import rospy
import os
import time
import json
import tyro
import threading
import numpy as np
import pandas as pd
import open3d as o3d
from termcolor import cprint
from dataclasses import dataclass
import matplotlib.pyplot as plt
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage, ShouldTerminateSensorMessage

from autolab_core import RigidTransform
from transforms3d.euler import euler2mat, mat2euler, euler2quat
from transforms3d.quaternions import mat2quat

from utils import get_traj, plot_trajectory
from robot_constants import TEST_JOINT_RANGE, TEST_EE_RANGE, REAL_K_GAINS, REAL_D_GAINS, INIT_GOAL_POSE

@dataclass
class Args:
    """Parameters for the EE pose data collection script for system identification."""

    show_plot: bool = False
    """Whether to display the generated trajectory plot."""

    trajectory_type: str = "sin"
    """The type of trajectory to generate (e.g., 'sin', 'step', 'valid', 'combined', 'policy'.)."""

    axis_idx: int = 0
    """The index of the axis to be identified (0-2: x,y,z translation, 3-5: rx,ry,rz rotation)."""

    ctrl_freq: int = 10
    """Control frequency in Hz."""

    output: str = "csv/"
    """The directory path for output data and parameters."""

    translation_range: float = 0.05
    """Maximum translation range for trajectory generation (meters)."""

    rotation_range: float = 0.2
    """Maximum rotation range for trajectory generation (radians)."""

class RealtimeEEPoseSysIdController:
    def __init__(self, state_update_freq=100, ctrl_freq=10):
        self.real_arm = FrankaArm('realtime_eepose_sysid_controller')
        
        # initialize lock and event for realtime control.
        self.current_pose = None
        self.state_lock = threading.Lock()
        self.stop_daemon_event = threading.Event()

        self.ctrl_rate = rospy.Rate(ctrl_freq)
        self.ctrl_duration = 1.0 / ctrl_freq
        self.state_update_rate = rospy.Rate(state_update_freq)
        
        vis_pose = False
        self.vis_pose = vis_pose

        self.state_updater_thread = threading.Thread(target=self._state_update_loop, daemon=True)
        rospy.loginfo("EE pose controller initialized.")

        if self.vis_pose:
            self.o3d_vis = o3d.visualization.Visualizer()
            self.o3d_vis.create_window()
            self.base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.o3d_vis.add_geometry(self.base_frame)

    def _state_update_loop(self):
        rospy.loginfo("daemon thread started, updating EE pose state...")
        while not self.stop_daemon_event.is_set():
            try:
                pose = self.real_arm.get_pose()

                with self.state_lock:
                    self.current_pose = pose
                
                self.state_update_rate.sleep()
      
            except rospy.ROSInterruptException:
                rospy.loginfo("ROS interrupted, stopping state update thread")
                break
            except Exception as e:
                rospy.logerr(f"daemon thread error: {e}")
                rospy.sleep(min(2.0, 0.1 * 2**getattr(self, '_error_count', 0)))
                self._error_count = getattr(self, '_error_count', 0) + 1

        rospy.loginfo("daemon thread stopped.")

    def get_latest_pose(self):
        with self.state_lock:
            retry_count = 0
            while self.current_pose is None:
                if retry_count > 50: 
                    rospy.logerr("Timeout waiting for pose state!")
                    raise TimeoutError("Failed to get pose state after multiple retries.")
                rospy.logwarn("waiting for the pose state first reading...")
                rospy.sleep(0.1)
                retry_count += 1
            return self.current_pose.copy()

    def _vis_pose(self, ee_pose):
        # update ee pose frame in Open3D viewer
        # ee_pose should be 4x4 transformation matrix
        self.ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.ee_frame.transform(ee_pose)
        self.o3d_vis.add_geometry(self.ee_frame)
        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()
        self.o3d_vis.remove_geometry(self.ee_frame)

    def get_pose_range(self):
        lower = TEST_EE_RANGE["lower"] # meters, radians
        upper = TEST_EE_RANGE["upper"] # meters, radians
        cprint(f"EE POSE limits lower : {lower}", "light_cyan", "on_blue")
        cprint(f"EE POSE limits upper : {upper}", "light_cyan", "on_blue")
        return lower, upper

    def start_backend_state_update(self):
        self.stop_daemon_event.clear()
        if hasattr(self, 'state_updater_thread') and self.state_updater_thread.is_alive():
            rospy.logwarn("State update thread already running")
            return
        if self.state_updater_thread._started.is_set():
            rospy.loginfo("Recreating thread object")
            self.state_updater_thread = threading.Thread(target=self._state_update_loop, daemon=True)
        self.state_updater_thread.start()
        rospy.loginfo("State update thread started")

    def stop(self):
        """ stop the daemon thread safely """
        rospy.loginfo("stopping the daemon thread...")
        self.stop_daemon_event.set()
        if self.state_updater_thread.is_alive():
            self.state_updater_thread.join(timeout=3)
            if self.state_updater_thread.is_alive():
                rospy.logwarn("Daemon thread did not stop in time, force stopping.")
        rospy.loginfo("daemon thread stopped.")

    def pose_to_vector(self, pose):
        """Convert pose to 6D vector [x, y, z, rx, ry, rz]"""
        # Extract position
        position = np.array([pose.translation[0], pose.translation[1], pose.translation[2]])
        
        # Extract rotation as euler angles
        rotation_matrix = pose.rotation
        # Convert rotation matrix to euler angles (rx, ry, rz)
        rx, ry, rz = mat2euler(rotation_matrix, 'sxyz')
        rotation = np.array([rx, ry, rz])
        
        return np.concatenate([position, rotation])

    def step_pose_action_real(self, target_pose_vec, init_time, param):
        '''
        Pass the pose action to real robot
        '''
        try:
            traj_gen_proto_msg = PosePositionSensorMessage(
                id=0, # maybe should be num of step
                timestamp=rospy.Time.now().to_time() - init_time,
                position=target_pose_vec[:3].tolist(),
                quaternion=euler2quat(*target_pose_vec[3:6], 'sxyz').tolist()
            )
            fb_ctrl_proto_msg = CartesianImpedanceSensorMessage(
                id=0, # maybe should be num of step
                timestamp=rospy.Time.now().to_time() - init_time,
                translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES,
                rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                feedback_controller_sensor_msg=sensor_proto2ros_msg(
                    fb_ctrl_proto_msg, SensorDataMessageType.CARTESIAN_IMPEDANCE),
            )
            self.real_arm.publish_sensor_values(ros_msg)
            cprint(f"Published real pose action: {target_pose_vec}", "green")
            rospy.loginfo('Publishing pose: ID {}'.format(traj_gen_proto_msg.id))
        except Exception as e:
            rospy.logerr(f"Failed to publish pose action: {e}")

    def run_trajectory_experiment(self, trajectory_type, axis_idx, param, show_plot=False, output="csv/", 
                                translation_range=0.05, rotation_range=0.2):
        try:
            self.start_backend_state_update()

            ''' -------------------------------------- GET COMMAND FOR SI -------------------------------------- '''

            defined_lower, defined_upper = self.get_pose_range()

            # Determine trajectory range based on axis
            if axis_idx < 3:  # Translation axes (x, y, z)
                center_value = (defined_lower[axis_idx] + defined_upper[axis_idx]) / 2
                lower = center_value - translation_range / 2
                upper = center_value + translation_range / 2                
            else:  # Rotation axes (rx, ry, rz)
                center_value = (defined_lower[axis_idx] + defined_upper[axis_idx]) / 2
                lower = center_value - rotation_range / 2
                upper = center_value + rotation_range / 2

            # Clamp to workspace limits
            lower = max(lower, defined_lower[axis_idx])
            upper = min(upper, defined_upper[axis_idx])

            trajectory_values = get_traj(trajectory_type, lower=lower, upper=upper)

            ''' -------------------------------------- MOVE TO INITIAL POSE -------------------------------------- '''
            # Move to initial pose
            self.real_arm.goto_joints(np.array(INIT_GOAL_POSE[:7]))
            self.real_arm.open_gripper()
            rospy.sleep(1.0)
            input("Hand on power switch... Press Enter to continue...")

            ''' -------------------------------------- INITIAL START DYNAMIC -------------------------------------- '''

            buffer_time = 20
            init_pose = self.get_latest_pose()
            if init_pose is None:
                rospy.logerr("Failed to get initial pose!")
                raise RuntimeError("Initial pose is None, cannot proceed with trajectory execution.")
            
            # Move to initial pose with impedance control
            self.real_arm.goto_pose(init_pose, 
                                  duration=20, 
                                  cartesian_impedances=param.get("cartesian_stiffness", FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES),
                                  dynamic=True, 
                                  buffer_time=buffer_time, 
                                  ignore_virtual_walls=True) 
            
            init_time = rospy.Time.now().to_time()
            start_time = time.time()
            
            # Initialize data recording lists
            goal_poses = []
            actual_poses = []
            t_history = []
            actual_vels = []

            # Convert initial pose to vector for trajectory generation
            init_pose_vector = self.pose_to_vector(init_pose)
            init_joint_state = self.real_arm.get_joints()

            ''' -------------------------------------- CONTROL LOOP -------------------------------------- '''
            
            # Realtime control loop
            for i, trajectory_value in enumerate(trajectory_values):
                init_time = rospy.Time.now().to_time() if i==0 else init_time

                # Create target pose by modifying only the specified axis
                target_pose_vector = init_pose_vector.copy()
                target_pose_vector[axis_idx] = np.clip(trajectory_value, lower, upper)

                # Send pose command (not blocking)
                self.step_pose_action_real(target_pose_vector, init_time, param)

                # Get current pose from daemon thread (not blocking)
                current_pose = self.get_latest_pose()
                if current_pose is None:
                    rospy.logerr(" ================= unable to acquire valid pose, skip recording this time ================= ")
                    continue
                
                # Data recording
                # Placeholder for actual velocity feedback
                signed_current_vel = 0  # TODO: implement velocity feedback if available
                
                goal_poses.append(target_pose_vector)
                actual_poses.append(self.pose_to_vector(current_pose))
                actual_vels.append(signed_current_vel)
                t_history.append(time.time() - start_time)

                # Visualize if enabled
                if self.vis_pose:
                    pose_matrix = np.eye(4)
                    pose_matrix[:3, :3] = current_pose.rotation
                    pose_matrix[:3, 3] = current_pose.translation
                    self._vis_pose(pose_matrix)

                # Strictly control frequency
                self.ctrl_rate.sleep()

            if not (len(actual_poses)==len(actual_vels)==len(t_history)==len(goal_poses)):
                rospy.loginfo("Data lengths do not match! Check your control loop.")

            rospy.loginfo("EE pose trajectory execution completed.")
            rospy.sleep(1.0)

            # Save data to csv
            axis_names = ['x', 'y', 'z', 'rx', 'ry', 'rz']
            df = pd.DataFrame({
                't': np.array(t_history),
                'Init qpos': np.array(init_joint_state).tolist(),
                'Goal Position': np.array(goal_poses).tolist(),
                'Actual Position': np.array(actual_poses).tolist(),
                'Actual Velocity': np.array(actual_vels),
            })
            filename = f"ee_pose_axis:{axis_names[axis_idx]}_trajType:{trajectory_type}"
            data_filename = f"{filename}.csv"
            os.makedirs(output, exist_ok=True)
            df.to_csv(os.path.join(output, data_filename), index=False)
            print(f"Data saved to {data_filename}")

            # Plot the trajectory for the specific axis
            goal_axis_values = [pose[axis_idx] for pose in goal_poses]
            actual_axis_values = [pose[axis_idx] for pose in actual_poses]

            plot_trajectory(t_history, actual_axis_values, goal_axis_values, actual_vels, 
                            filename, output, show_plot)
            print(f"Time taken: {time.time() - start_time}s")

        except KeyboardInterrupt:
            rospy.loginfo("Experiment interrupted by user")
        except Exception as e:
            rospy.logerr(f"Experiment failed with error: {e}")
            raise
        finally:
            self.real_arm.stop_skill()
            self.stop()
            rospy.loginfo("EE pose system identification experiment completed.")


def main(args: Args):
    """Main execution function that accepts an instance of the Args class."""
    cprint(f"Starting EE pose system identification trajectory generation...\n{args}", "yellow")
    cprint(args, "yellow")
    
    param = {
        "cartesian_stiffness": FC.DEFAULT_TRANSLATIONAL_STIFFNESSES + FC.DEFAULT_ROTATIONAL_STIFFNESSES,
        "cartesian_damping": FC.DEFAULT_TRANSLATIONAL_DAMPINGS + FC.DEFAULT_ROTATIONAL_DAMPINGS,
    }
    
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "real_eepose_param.json"), "w") as f:
        json.dump({k: v.tolist() if hasattr(v, 'tolist') else v for k, v in param.items()}, f, indent=4)

    controller = RealtimeEEPoseSysIdController(ctrl_freq=args.ctrl_freq)
    controller.run_trajectory_experiment(
        trajectory_type=args.trajectory_type,
        axis_idx=args.axis_idx,
        param=param,
        show_plot=args.show_plot,
        output=args.output,
        translation_range=args.translation_range,
        rotation_range=args.rotation_range
    )

if __name__ == "__main__":
    tyro.cli(main)