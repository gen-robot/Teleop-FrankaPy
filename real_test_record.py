#!/usr/bin/env python3
"""
Generate three simple robot episodes that mimic the Phase 0 style movement
(moving from the initial pose to a new target pose) using the same control
patterns, timing, and data recording structure as the pull_drawer dataset.

Episodes:
- Episode 0: +0.10 m along X from initial pose
- Episode 1: +0.10 m along Y from initial pose
- Episode 2: +0.10 m along Z from initial pose

Data collection mirrors VLADataCollector format used by pull_drawer scripts,
but only robot data is recorded (no images/videos).

A 3D plot with the three trajectories and TCP orientation arrows is saved
alongside the dataset.
"""

import os
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plotting)

# Franka/Robotics imports
from autolab_core import RigidTransform
from transforms3d.euler import euler2quat, euler2mat, mat2euler
from frankapy import FrankaArm, SensorDataMessageType
from frankapy.franka_constants import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage

# Data collection (same structure as pull_drawer)
from examples.data_collection.vla_data_collector import VLADataCollector
from realsense_wrapper.realsense_d435 import RealsenseAPI

import rospy


class RealTestRecorder:
    def __init__(self, dataset_dir: str, control_frequency: int = 5, duration: float = 1.5, debug: bool = False, dry_run: bool = False):
        self.dataset_dir = dataset_dir
        self.control_frequency = control_frequency  # same as reference
        self.duration = duration  # seconds, same as Phase 0 in reference
        self.debug = debug
        self.dry_run = dry_run

        os.makedirs(self.dataset_dir, exist_ok=True)

        if self.dry_run:
            # Use mock objects for a dry run to avoid hardware dependencies
            class MockRobot:
                def get_state(self): return {
                    'joint_positions': np.zeros(7), 'joint_velocities': np.zeros(7),
                    'cartesian_pose': np.identity(4),
                    'cartesian_velocities': {'linear': np.zeros(3), 'angular': np.zeros(3)},
                    'gripper_width': FC.GRIPPER_WIDTH_MAX, 'gripper_is_grasped': False,
                }
            class MockCameras:
                def get_intrinsics(self): return {}
                def get_image_data(self): return {}, 0.0

            self.robot = MockRobot()
            self.cameras = MockCameras()
        else:
            # Initialize robot and data collector with cameras for image recording
            self.robot = FrankaArm()
            self.cameras = RealsenseAPI()

        self.data_collector = VLADataCollector(self.robot, cameras=self.cameras)

        # Execution state
        self.init_xyz = None
        self.init_rotation = None
        self.command_xyz = None
        self.command_rotation = None

    # ---- Helper methods ported from reference script ----
    def _ee_pose_init(self):
        """Establish baseline pose for delta accumulation (like data_replayer)."""
        if not self.dry_run:
            time.sleep(0.5)
            pose = self.robot.get_pose()
        else:
            # For a dry run, use a fixed, predictable starting pose
            pose = FC.HOME_POSE

        self.init_xyz = pose.translation
        self.init_rotation = pose.rotation
        self.command_xyz = self.init_xyz.copy()
        self.command_rotation = self.init_rotation.copy()
        print(f"Initialized EE pose tracking at: {self.command_xyz}")

    def _ensure_action_client_ready(self):
        """Simplified readiness check (mirrors reference approach)."""
        if self.dry_run: return
        print("[INFO] Ensuring robot is ready for new episode...")
        max_wait_time = 3.0
        start = time.time()
        while time.time() - start < max_wait_time:
            try:
                franka_status = self.robot._get_current_franka_interface_status().franka_interface_status
                if franka_status.is_ready and not self.robot._in_skill:
                    print("[INFO] Robot confirmed ready for new episode")
                    time.sleep(0.2)
                    return
            except Exception as e:
                print(f"[WARN] Error checking robot status: {e}")
            time.sleep(0.1)
        print("[WARN] Robot readiness timeout, proceeding with care...")
        time.sleep(1.0)

    def _stop_skill_safely(self):
        """Safe stop to avoid PREEMPTING/DONE race condition (from reference)."""
        if self.dry_run: return
        import time as _t
        print("[INFO] Safely stopping skill...")
        if not self.robot._connected:
            print("[INFO] Robot not connected")
            return
        try:
            client_state = self.robot._client.get_state()
            if client_state in (3, 4) or not self.robot._in_skill:
                self.robot._in_skill = False
                print("[INFO] No active skill or already completed")
                return
        except Exception as e:
            print(f"[DEBUG] Could not check action client state: {e}")
            if not self.robot._in_skill:
                return
        print("[INFO] Cancelling current goal...")
        try:
            self.robot._client.cancel_goal()
        except Exception as e:
            print(f"[DEBUG] Error during goal cancellation: {e}")
        max_wait_time = 3.0
        start = _t.time()
        print("[INFO] Waiting for skill cancellation to complete...")
        while _t.time() - start < max_wait_time:
            try:
                franka_status = self.robot._get_current_franka_interface_status().franka_interface_status
                if franka_status.is_ready:
                    self.robot._in_skill = False
                    print("[INFO] Skill safely stopped (franka interface ready)")
                    return
            except Exception as e:
                print(f"[WARN] Error checking franka status: {e}")
            _t.sleep(0.05)
        print("[WARN] Skill stop timeout, forcing state reset...")
        self.robot._in_skill = False

    def robot_init_for_episode(self):
        """Reset joints, open gripper, goto HOME_POSE (dynamic=True), then init pose baseline."""
        print("Initializing robot for new episode...")
        if not self.dry_run:
            self._ensure_action_client_ready()
            time.sleep(0.5)
            print("Resetting joints...")
            # Retry guard similar to reference
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    franka_status = self.robot._get_current_franka_interface_status().franka_interface_status
                    if not franka_status.is_ready or self.robot._in_skill:
                        print(f"[WARN] Robot not ready on attempt {attempt+1}, waiting...")
                        time.sleep(1.0)
                        continue
                    self.robot.reset_joints()
                    print("Joint reset completed successfully")
                    break
                except ValueError as e:
                    if "Cannot send another command" in str(e) and attempt < max_attempts - 1:
                        print(f"[WARN] Command conflict: {e}. Retrying...")
                        time.sleep(2.0)
                        continue
                    raise
            print("Opening gripper...")
            self.robot.open_gripper()
            print("Moving to HOME_POSE...")
            self.robot.goto_pose(
                FC.HOME_POSE,
                duration=10,
                dynamic=True,
                buffer_time=100000000,
                skill_desc='MOVE',
                cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES,
                ignore_virtual_walls=True,
            )
            print("HOME_POSE movement completed.")
        else:
            print("[DRY RUN] Skipping physical robot initialization.")

        self._ee_pose_init()
        print("Robot initialized and ready for episode.")

    # ---- Trajectory generation and execution ----
    def _interpolate_positions(self, waypoints, num_points):
        if len(waypoints) < 2:
            return np.array(waypoints)
        waypoints = np.array(waypoints)
        t_waypoints = np.linspace(0, 1, len(waypoints))
        t_interp = np.linspace(0, 1, num_points)
        interp_x = np.interp(t_interp, t_waypoints, waypoints[:, 0])
        interp_y = np.interp(t_interp, t_waypoints, waypoints[:, 1])
        interp_z = np.interp(t_interp, t_waypoints, waypoints[:, 2])
        return np.column_stack([interp_x, interp_y, interp_z])

    def _generate_linear_axis_move(self, axis_offset: np.ndarray):
        """Make a Phase-0-like straight-line path from init pose to init+offset."""
        start_pose = RigidTransform(rotation=self.init_rotation, translation=self.init_xyz,
                                    from_frame='franka_tool', to_frame='world')
        target_pose = RigidTransform(rotation=self.init_rotation,  # keep orientation
                                     translation=self.init_xyz + axis_offset,
                                     from_frame='franka_tool', to_frame='world')
        num_steps = int(self.duration * self.control_frequency)
        positions = self._interpolate_positions([start_pose.translation, target_pose.translation], num_steps)
        # Keep orientation constant; delta eulers will be zeros
        delta_positions = np.diff(positions, axis=0, prepend=positions[0:1])
        delta_eulers = np.zeros_like(delta_positions)
        grippers = np.full(num_steps, 1.0)  # open
        return delta_positions, delta_eulers, grippers, positions

    def _generate_orientation_move(self, euler_offset: np.ndarray):
        """Make orientation-only movement from init pose to init+euler_offset."""
        start_pose = RigidTransform(rotation=self.init_rotation, translation=self.init_xyz,
                                    from_frame='franka_tool', to_frame='world')
        # Apply euler offset to rotation
        offset_rotation = euler2mat(euler_offset[0], euler_offset[1], euler_offset[2], 'sxyz')
        target_rotation = np.matmul(self.init_rotation, offset_rotation)
        target_pose = RigidTransform(rotation=target_rotation, translation=self.init_xyz,  # keep position
                                     from_frame='franka_tool', to_frame='world')
        num_steps = int(self.duration * self.control_frequency)
        positions = self._interpolate_positions([start_pose.translation, target_pose.translation], num_steps)
        # Generate orientation deltas
        orientations = self._interpolate_orientations([start_pose.quaternion, target_pose.quaternion], num_steps)
        delta_positions = np.diff(positions, axis=0, prepend=positions[0:1])
        delta_eulers = self._quaternions_to_delta_eulers(orientations)
        grippers = np.full(num_steps, 1.0)  # open
        return delta_positions, delta_eulers, grippers, positions, orientations

    def _execute_delta_trajectory(self, delta_positions, delta_eulers, grippers, record_instruction: str):
        """Execute deltas using data-replayer methodology and record robot-only data."""
        print(f"Executing trajectory with {len(delta_positions)} steps...")
        if not self.dry_run:
            # CRITICAL FIX: Stop any previous skill before starting new trajectory (like reference script)
            print("[INFO] Stopping any previous skill before starting trajectory...")
            self.robot.stop_skill()

            # Start dynamic skill at current pose
            current_pose = self.robot.get_pose()
            self.robot.goto_pose(
                current_pose,
                duration=10,
                dynamic=True,
                buffer_time=100000000,
                skill_desc='REAL_TEST_TRAJECTORY',
                cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES,
                ignore_virtual_walls=True,
            )
            time.sleep(FC.DYNAMIC_SKILL_WAIT_TIME)

        if not self.dry_run:
            rate = rospy.Rate(self.control_frequency)
            self.init_time = rospy.Time.now().to_time()
        else:
            rate = None
            self.init_time = time.time()

        self.data_collector.clear_data()
        self.command_xyz = self.init_xyz.copy()
        self.command_rotation = self.init_rotation.copy()

        for step in range(len(delta_positions)):
            try:
                if not self.dry_run:
                    timestamp = rospy.Time.now().to_time() - self.init_time
                else:
                    timestamp = time.time() - self.init_time
                dxyz = delta_positions[step]
                deuler = delta_eulers[step]
                grw = grippers[step]

                delta_rotation = euler2mat(deuler[0], deuler[1], deuler[2], 'sxyz')
                self.command_xyz += dxyz
                self.command_rotation = np.matmul(self.command_rotation, delta_rotation)

                if not self.dry_run:
                    cmd_tf = RigidTransform(
                        rotation=self.command_rotation,
                        translation=self.command_xyz,
                        from_frame='franka_tool',
                        to_frame='world',
                    )
                    gripper_width_scaled = FC.GRIPPER_WIDTH_MAX * grw

                    pub_traj_gen_proto_msg = PosePositionSensorMessage(
                        id=step+1, timestamp=timestamp,
                        position=cmd_tf.translation,
                        quaternion=cmd_tf.quaternion,
                    )
                    fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                        id=step+1, timestamp=timestamp,
                        translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES,
                        rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES,
                    )
                    ros_pub_sensor_msg = make_sensor_group_msg(
                        trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                            pub_traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                        feedback_controller_sensor_msg=sensor_proto2ros_msg(
                            fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE),
                    )
                    self.robot.publish_sensor_values(ros_pub_sensor_msg)

                    # Gripper control (stays open here but keep logic consistent)
                    current_grw = self.robot.get_gripper_width()
                    if abs(gripper_width_scaled - current_grw) > 0.01:
                        grasp = True if grw < 0.5 else False
                        self.robot.goto_gripper(gripper_width_scaled, grasp=grasp,
                                                force=FC.GRIPPER_MAX_FORCE/3.0, speed=0.12,
                                                block=False, skill_desc="control_gripper")

                # Record robot-only data
                save_action = {
                    "delta": {
                        "position": dxyz,
                        "orientation": euler2quat(deuler[0], deuler[1], deuler[2], 'sxyz'),
                        "euler_angle": deuler,
                    },
                    "abs": {
                        "position": self.command_xyz.copy(),
                        "euler_angle": np.array([mat2euler(self.command_rotation, 'sxyz')])[0],
                    },
                    "gripper_width": grw,
                }
                self.data_collector.update_data_dict(instruction=record_instruction, action=save_action, timestamp=timestamp)

            except Exception as e:
                # Simple recovery: reinitialize baseline and continue
                self._ee_pose_init()
                if not self.dry_run:
                    rate.sleep()
                else:
                    time.sleep(1.0 / self.control_frequency)
                print(f"[WARN] Move failed? : {e}")
                continue

            if not self.dry_run:
                rate.sleep()
            else:
                time.sleep(1.0 / self.control_frequency)

        if not self.dry_run:
            print("Trajectory execution finished. Stopping dynamic skill...")
            self._stop_skill_safely()
        else:
            print("[DRY RUN] Trajectory execution finished.")

    def _save_episode(self, episode_idx: int, meta: dict):
        episode_dir = os.path.join(self.dataset_dir, f"episode_{episode_idx}")
        os.makedirs(episode_dir, exist_ok=True)
        # Save without videos/images on a dry run
        is_save_video = not self.dry_run
        self.data_collector.save_data(episode_dir, episode_idx, is_compressed=False, is_save_video=is_save_video)
        metadata_path = os.path.join(episode_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=4)
        print(f"Episode {episode_idx} saved at {episode_dir}")

    def _generate_orientation_move(self, euler_offset: np.ndarray):
        """Make orientation-only movement from init pose to init+euler_offset."""
        start_pose = RigidTransform(rotation=self.init_rotation, translation=self.init_xyz,
                                    from_frame='franka_tool', to_frame='world')
        # Apply euler offset to rotation
        offset_rotation = euler2mat(euler_offset[0], euler_offset[1], euler_offset[2], 'sxyz')
        target_rotation = np.matmul(self.init_rotation, offset_rotation)
        target_pose = RigidTransform(rotation=target_rotation, translation=self.init_xyz,  # keep position
                                     from_frame='franka_tool', to_frame='world')
        num_steps = int(self.duration * self.control_frequency)
        positions = self._interpolate_positions([start_pose.translation, target_pose.translation], num_steps)
        # Generate orientation deltas
        orientations = self._interpolate_orientations([start_pose.quaternion, target_pose.quaternion], num_steps)
        delta_positions = np.diff(positions, axis=0, prepend=positions[0:1])
        delta_eulers = self._quaternions_to_delta_eulers(orientations)
        grippers = np.full(num_steps, 1.0)  # open
        return delta_positions, delta_eulers, grippers, positions, orientations

    def _interpolate_orientations(self, quaternions, num_points):
        """Interpolate orientations using proper SLERP"""
        if len(quaternions) < 2:
            return np.array(quaternions)
        t_waypoints = np.linspace(0, 1, len(quaternions))
        t_interp = np.linspace(0, 1, num_points)
        result = []
        for t in t_interp:
            idx = np.searchsorted(t_waypoints, t) - 1
            idx = max(0, min(idx, len(quaternions) - 2))
            if t_waypoints[idx + 1] == t_waypoints[idx]:
                local_t = 0
            else:
                local_t = (t - t_waypoints[idx]) / (t_waypoints[idx + 1] - t_waypoints[idx])
            q_interp = self._slerp(quaternions[idx], quaternions[idx + 1], local_t)
            result.append(q_interp)
        return np.array(result)

    def _slerp(self, q1, q2, t):
        """Spherical linear interpolation for quaternions"""
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            result = q1 * (1 - t) + q2 * t
            return result / np.linalg.norm(result)
        theta = np.arccos(np.abs(dot))
        sin_theta = np.sin(theta)
        factor1 = np.sin((1 - t) * theta) / sin_theta
        factor2 = np.sin(t * theta) / sin_theta
        return (factor1 * q1 + factor2 * q2) / np.linalg.norm(factor1 * q1 + factor2 * q2)

    def _quaternions_to_delta_eulers(self, quaternions):
        """Convert quaternion sequence to delta euler sequence"""
        eulers = np.array([mat2euler(RigidTransform(rotation=q, translation=[0,0,0]).rotation, 'sxyz')
                          for q in quaternions])
        return np.diff(eulers, axis=0, prepend=eulers[0:1])

    def _precompute_axis_sequences(self, offset_m: float = 0.1):
        """Pre-generate linear trajectories for X/Y/Z and orientation moves from the current baseline."""
        pre = {}
        deg2rad = np.pi / 180
        euler_90deg = 45.0 * deg2rad  # 90 degrees in positive direction

        mapping = {
            "X": (0, np.array([offset_m, 0.0, 0.0]), None),
            "Y": (1, np.array([0.0, offset_m, 0.0]), None),
            "Z": (2, np.array([0.0, 0.0, offset_m]), None),
            "Roll": (3, None, np.array([euler_90deg, 0.0, 0.0])),
            "Pitch": (4, None, np.array([0.0, euler_90deg, 0.0])),
            "Yaw": (5, None, np.array([0.0, 0.0, euler_90deg])),
        }

        for axis_name, (ep_idx, pos_offset, euler_offset) in mapping.items():
            if pos_offset is not None:
                # Position movement
                dpos, deuler, gr, positions = self._generate_linear_axis_move(pos_offset)
                rotations = np.array([self.init_rotation for _ in range(len(positions))])
            else:
                # Orientation movement
                dpos, deuler, gr, positions, orientations = self._generate_orientation_move(euler_offset)
                rotations = np.array([RigidTransform(rotation=q, translation=[0,0,0]).rotation for q in orientations])

            pre[axis_name] = {
                "episode_idx": ep_idx,
                "delta_positions": dpos,
                "delta_eulers": deuler,
                "grippers": gr,
                "positions": positions,
                "rotations": rotations,
            }
        return pre

    def run_selected_axes(self, offset_m: float = 0.1, do_x: bool = False, do_y: bool = False, do_z: bool = False,
                         do_roll: bool = False, do_pitch: bool = False, do_yaw: bool = False, save_plot_name: str = "trajectories.png"):
        # 1) Initialize once and precompute all trajectories from this baseline
        self.robot_init_for_episode()
        pre = self._precompute_axis_sequences(offset_m=offset_m)

        # 2) Prepare plot data for all axes regardless of selection
        all_trajs = []
        for axis_name in ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]:
            item = pre[axis_name]
            all_trajs.append({
                "name": f"Episode {item['episode_idx']} ({axis_name})",
                "positions": item["positions"],
                "rotations": item["rotations"],
            })

        # 3) Decide which axes to execute; default to position axes if none specified
        selection = []
        if do_x: selection.append("X")
        if do_y: selection.append("Y")
        if do_z: selection.append("Z")
        if do_roll: selection.append("Roll")
        if do_pitch: selection.append("Pitch")
        if do_yaw: selection.append("Yaw")
        if not selection:
            selection = ["X", "Y", "Z"]

        # 4) Execute selected axes; reinitialize between axes to ensure consistent episodes
        for idx, axis_name in enumerate(selection):
            item = pre[axis_name]
            ep_idx = item["episode_idx"]
            print(f"\n=== Episode {ep_idx}: +{offset_m} m along {axis_name} ===")

            # CRITICAL FIX: Stop any active skill before reinitializing (like reference script)
            if idx > 0 and not self.dry_run:
                print("[INFO] Stopping previous skill before reinitializing...")
                self.robot.stop_skill()
                self.robot_init_for_episode()

            self._execute_delta_trajectory(item["delta_positions"], item["delta_eulers"], item["grippers"], record_instruction=f"real_test_axis_{axis_name.lower()}")
            steps_recorded = len(self.data_collector.data_dict['action']['end_effector']['delta_position'])
            meta = {
                "task_name": "real_test_record",
                "episode_idx": ep_idx,
                "axis": axis_name,
                "offset_m": float(offset_m),
                "action_steps": steps_recorded,
                "generation_method": "phase0_style_linear_move",
            }
            self._save_episode(ep_idx, meta)

        # 5) Plot all three trajectories
        self._plot_trajectories(all_trajs, os.path.join(self.dataset_dir, save_plot_name))

    def _plot_trajectories(self, trajs, save_path):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        colors = ["r", "g", "b"]
        for i, t in enumerate(trajs):
            pos = t["positions"]
            rot = t["rotations"]  # [N,3,3]
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=colors[i % len(colors)], label=t["name"], linewidth=2)
            # Orientation arrows (TCP x-axis shown)
            step_idx = list(range(0, len(pos), max(1, len(pos)//5)))
            for j in step_idx:
                R = rot[j]
                origin = pos[j]
                x_axis = R[:, 0] * 0.05  # short arrow
                ax.quiver(origin[0], origin[1], origin[2], x_axis[0], x_axis[1], x_axis[2],
                          color=colors[i % len(colors)], length=0.05, normalize=False)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(loc='best')
        ax.set_title('Real Test Trajectories (Position + Orientation)')
        ax.view_init(elev=25, azim=-60)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Saved trajectory plot to: {save_path}")


def get_arguments():
    parser = argparse.ArgumentParser(description="Generate axis linear episodes with Phase-0-like motion and normal image+robot data.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/yukun/real_test_record", help="Directory to save episodes.")
    parser.add_argument("--offset", type=float, default=0.1, help="Axis offset in meters for each episode (default: 0.1)")
    parser.add_argument("--control_frequency", type=int, default=5, help="Control frequency in Hz (default: 5)")
    parser.add_argument("--duration", type=float, default=1.5, help="Move duration in seconds (default: 1.5)")
    parser.add_argument("--x", action="store_true", help="Execute X axis episode")
    parser.add_argument("--y", action="store_true", help="Execute Y axis episode")
    parser.add_argument("--z", action="store_true", help="Execute Z axis episode")
    parser.add_argument("--roll", action="store_true", help="Execute Roll orientation episode (+90 deg)")
    parser.add_argument("--pitch", action="store_true", help="Execute Pitch orientation episode (+90 deg)")
    parser.add_argument("--yaw", action="store_true", help="Execute Yaw orientation episode (+90 deg)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("--dry_run", action="store_true", help="Skip robot communication for a dry run, saving simulated data and plots.")
    return parser.parse_args()


def main():
    args = get_arguments()
    print("=== Real Test Recorder ===")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Offset: {args.offset} m")
    print(f"Control frequency: {args.control_frequency} Hz, Duration: {args.duration} s")
    if args.dry_run:
        print("\n[WARN] DRY RUN MODE IS ENABLED. NO ROBOT COMMANDS WILL BE SENT.\n")


    rec = RealTestRecorder(dataset_dir=args.dataset_dir,
                           control_frequency=args.control_frequency,
                           duration=args.duration,
                           debug=args.debug,
                           dry_run=args.dry_run)
    rec.run_selected_axes(offset_m=args.offset, do_x=args.x, do_y=args.y, do_z=args.z,
                         do_roll=args.roll, do_pitch=args.pitch, do_yaw=args.yaw, save_plot_name="trajectories.png")


if __name__ == "__main__":
    main()