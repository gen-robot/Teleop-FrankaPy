#!/usr/bin/env python3
"""
Three-phase trajectory generation script for extending the pull_drawer dataset.

Phase 0: Initialization movement (NOT recorded) - move from HOME_POSE to random pose
Phase 1: Connection movement (recorded) - move from random pose to base episode start
Phase 2: Execute base episode from selected starting pose

Usage:
    python3 generate_pull_drawer_data.py --num_episodes 5 --base_episode 13   --pos_x_range -0.0 0.1 --pos_y_range -0.10 0.10 --pos_z_range 0.0 0.0
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import copy
from scipy.interpolate import CubicSpline
from transforms3d.euler import euler2quat, euler2mat, mat2euler

# FrankaPy imports
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy.franka_constants import FrankaConstants as FC
from realsense_wrapper.realsense_d435 import RealsenseAPI
from examples.data_collection.vla_data_collector import VLADataCollector
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage
import rospy


class ThreePhaseDataGenerator:
    def __init__(self, base_dataset_dir="pull_drawer", new_dataset_dir="pull_drawer_new",
                 pos_x_range=None, pos_y_range=None, pos_z_range=None):
        """
        Initialize the three-phase data generator.

        Args:
            base_dataset_dir: Directory containing the original pull_drawer dataset
            new_dataset_dir: Directory to save new generated episodes
            pos_x_range: Position offset range for X axis [min, max] in meters
            pos_y_range: Position offset range for Y axis [min, max] in meters
            pos_z_range: Position offset range for Z axis [min, max] in meters
        """
        self.base_dataset_dir = base_dataset_dir
        self.new_dataset_dir = new_dataset_dir
        self.control_frequency = 5  # Hz, same as original data collection
        
        # Initialize robot and cameras
        # FrankaArm will handle ROS node initialization automatically
        self.robot = FrankaArm()
        self.cameras = RealsenseAPI()
        self.data_collector = VLADataCollector(self.robot, self.cameras)
        
        # Load base dataset
        self.base_episodes = self._load_base_episodes()
        
        # Create new dataset directory
        os.makedirs(self.new_dataset_dir, exist_ok=True)
        
        # Random pose delta limits (relative to current position)
        # Position deltas: configurable per axis
        # Orientation deltas: -10 to 10 degrees in each axis
        deg2rad = np.pi / 180  # Easy to change: modify degrees above, conversion happens here
        self.random_delta_limits = {
            'position_x': pos_x_range if pos_x_range is not None else [-0.05, 0.05],
            'position_y': pos_y_range if pos_y_range is not None else [-0.05, 0.05],
            'position_z': pos_z_range if pos_z_range is not None else [-0.05, 0.05],
            'orientation': [-10 * deg2rad, 10 * deg2rad]  # ±10 degrees in each axis
        }
        
    def _load_base_episodes(self):
        """Load all base episodes from the original dataset."""
        base_episodes = {}
        episode_dirs = [d for d in os.listdir(self.base_dataset_dir) 
                       if d.startswith('episode_') and os.path.isdir(os.path.join(self.base_dataset_dir, d))]
        
        for episode_dir in episode_dirs:
            episode_idx = int(episode_dir.split('_')[1])
            data_path = os.path.join(self.base_dataset_dir, episode_dir, 'data.npy')
            
            if os.path.exists(data_path):
                try:
                    episode_data = np.load(data_path, allow_pickle=True).item()
                    base_episodes[episode_idx] = episode_data
                    print(f"Loaded base episode {episode_idx}")
                except Exception as e:
                    print(f"Failed to load episode {episode_idx}: {e}")
                    
        return base_episodes
    
    def _get_next_episode_idx(self):
        """Get the next episode index for the new dataset."""
        if not os.path.exists(self.new_dataset_dir):
            return 15  # Start from 15 since base dataset goes to 14
            
        existing_episodes = [d for d in os.listdir(self.new_dataset_dir) 
                           if d.startswith('episode_') and os.path.isdir(os.path.join(self.new_dataset_dir, d))]
        
        if not existing_episodes:
            return 15
            
        max_idx = max([int(d.split('_')[1]) for d in existing_episodes])
        return max_idx + 1
    
    def _generate_random_pose(self):
        """
        Generate a random target pose using fixed offset from initial pose.
        Position offset: -0.05 to 0.05m, Orientation offset: -10 to 10 degrees.

        Returns:
            RigidTransform: Target pose = initial_pose + random_offset
        """
        # Use initial pose as baseline (set by _ee_pose_init after robot initialization)
        initial_position = self.init_xyz
        initial_rotation = self.init_rotation

        # Generate random position offset within limits (per axis)
        position_offset = np.array([
            np.random.uniform(self.random_delta_limits['position_x'][0], self.random_delta_limits['position_x'][1]),
            np.random.uniform(self.random_delta_limits['position_y'][0], self.random_delta_limits['position_y'][1]),
            np.random.uniform(self.random_delta_limits['position_z'][0], self.random_delta_limits['position_z'][1])
        ])

        # Generate random orientation offset within limits
        orientation_offset = np.array([
            np.random.uniform(self.random_delta_limits['orientation'][0], self.random_delta_limits['orientation'][1]),
            np.random.uniform(self.random_delta_limits['orientation'][0], self.random_delta_limits['orientation'][1]),
            np.random.uniform(self.random_delta_limits['orientation'][0], self.random_delta_limits['orientation'][1])
        ])

        # Apply offset to initial pose
        target_position = initial_position + position_offset

        # Apply orientation offset
        offset_rotation = euler2mat(orientation_offset[0], orientation_offset[1], orientation_offset[2], 'sxyz')
        target_rotation = np.matmul(initial_rotation, offset_rotation)

        # Create target pose
        target_pose = RigidTransform(
            rotation=target_rotation,
            translation=target_position,
            from_frame='franka_tool',
            to_frame='world'
        )

        # Convert orientation offset back to degrees for logging
        deg2rad = np.pi / 180
        orientation_offset_deg = orientation_offset / deg2rad

        print(f"Generated random offset: pos={position_offset}, ori_deg={orientation_offset_deg}")
        print(f"Target pose: pos={target_position}, from initial pos={initial_position}")
        return target_pose

    def _slerp(self, q1, q2, t):
        """
        Proper spherical linear interpolation (SLERP) for quaternions.

        Args:
            q1: Start quaternion (numpy array)
            q2: End quaternion (numpy array)
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated quaternion (numpy array)
        """
        # Ensure quaternions are normalized
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)

        # Compute dot product
        dot = np.dot(q1, q2)

        # If dot product is negative, use -q2 to ensure shortest path
        if dot < 0:
            q2 = -q2
            dot = -dot

        # If quaternions are very close, use linear interpolation to avoid numerical issues
        if dot > 0.9995:
            result = q1 * (1 - t) + q2 * t
            return result / np.linalg.norm(result)

        # Calculate angle between quaternions
        theta = np.arccos(np.abs(dot))
        sin_theta = np.sin(theta)

        # Compute SLERP
        factor1 = np.sin((1 - t) * theta) / sin_theta
        factor2 = np.sin(t * theta) / sin_theta

        result = factor1 * q1 + factor2 * q2
        return result / np.linalg.norm(result)

    def _interpolate_trajectory(self, start_pose, end_pose, duration, num_points=None):
        """
        Generate smooth interpolated trajectory between two poses using cubic splines.

        Args:
            start_pose: Starting RigidTransform
            end_pose: Ending RigidTransform
            duration: Duration of trajectory in seconds
            num_points: Number of interpolation points (if None, calculated from duration and frequency)

        Returns:
            List of RigidTransform poses along the trajectory
        """
        if num_points is None:
            num_points = max(int(duration * self.control_frequency), 2)

        # Time points
        t = np.linspace(0, 1, num_points)

        # Interpolate position using cubic spline
        pos_spline = CubicSpline([0, 1], [start_pose.translation, end_pose.translation])
        positions = pos_spline(t)

        # Interpolate orientation using proper SLERP (spherical linear interpolation)
        start_quat = start_pose.quaternion
        end_quat = end_pose.quaternion

        # Ensure shortest path interpolation
        if np.dot(start_quat, end_quat) < 0:
            end_quat = -end_quat

        trajectory = []
        for i, pos in enumerate(positions):
            # FIXED: Proper SLERP implementation instead of linear interpolation
            alpha = t[i]
            quat = self._slerp(start_quat, end_quat, alpha)

            pose = RigidTransform(
                rotation=quat,
                translation=pos,
                from_frame='franka_tool',
                to_frame='world'
            )
            trajectory.append(pose)

        # VERIFICATION: Check end-point accuracy
        final_pose = trajectory[-1]
        pos_error = np.linalg.norm(final_pose.translation - end_pose.translation)

        # Compare quaternions (accounting for q and -q representing same rotation)
        quat_error = min(
            np.linalg.norm(final_pose.quaternion - end_pose.quaternion),
            np.linalg.norm(final_pose.quaternion + end_pose.quaternion)
        )

        if pos_error > 1e-6:
            print(f"[WARN] Trajectory position error: {pos_error:.8f}m")
        if quat_error > 1e-6:
            print(f"[WARN] Trajectory orientation error: {quat_error:.8f}")

        return trajectory

    def _ee_pose_init(self):
        """
        Initialize end-effector pose tracking (like data_replayer.py).
        This establishes the current pose as the baseline for delta action accumulation.
        """
        time.sleep(0.5)
        pose = self.robot.get_pose()
        self.init_xyz = pose.translation
        self.init_rotation = pose.rotation
        self.command_xyz = self.init_xyz.copy()
        self.command_rotation = self.init_rotation.copy()
        print(f"Initialized EE pose tracking at: {self.command_xyz}")

    def _execute_phase0_initialization(self, target_pose, duration=5.0):
        """
        Phase 0: Execute smooth movement to random pose WITHOUT recording data.
        This is the initialization phase that moves from HOME_POSE to random pose.
        Uses delta action interpolation with HOME_POSE as baseline (established at start).

        Args:
            target_pose: Target RigidTransform to move to
            duration: Duration of movement in seconds
        """
        print(f"Phase 0: Moving to random pose (initialization, not recorded)...")
        print(f"Target position: {target_pose.translation}")
        print(f"Target orientation (euler): {mat2euler(target_pose.rotation, 'sxyz')}")

        # CRITICAL: Initialize pose tracking baseline at HOME_POSE (start of Phase 0)
        # This establishes HOME_POSE as the baseline for ALL subsequent delta accumulation
        self._ee_pose_init()
        print(f"Phase 0: Established HOME_POSE baseline at {self.command_xyz}")

        # Ensure dynamic skill is active for delta action execution
        current_pose = self.robot.get_pose()
        self.robot.goto_pose(current_pose, duration=10, dynamic=True,
                           buffer_time=100000000, skill_desc='PHASE0_INITIALIZATION',
                           cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES,
                           ignore_virtual_walls=True)

        # Wait for dynamic skill to initialize
        time.sleep(FC.DYNAMIC_SKILL_WAIT_TIME)

        # Generate delta action sequence to reach target pose
        delta_actions = self._generate_delta_action_sequence(target_pose, duration)

        # Execute delta actions WITHOUT data recording (key difference from Phase 1)
        self._execute_delta_actions_without_recording(delta_actions, "Phase 0: Initialization movement")

        # Stop the dynamic skill after phase 0
        self.robot.stop_skill()

        print("Phase 0 completed successfully")

    def _generate_delta_action_sequence(self, target_pose, duration):
        """
        Generate a sequence of delta actions to reach target pose (like original data collection).

        Args:
            target_pose: Target RigidTransform to reach
            duration: Duration of movement in seconds

        Returns:
            List of delta actions (position and euler deltas)
        """
        # Calculate total displacement needed
        current_pose = self.robot.get_pose()
        total_position_delta = target_pose.translation - current_pose.translation

        # Calculate rotation delta
        current_rotation = current_pose.rotation
        target_rotation = target_pose.rotation
        relative_rotation = target_rotation @ current_rotation.T
        total_euler_delta = np.array(mat2euler(relative_rotation, 'sxyz'), dtype=np.float64)

        # Generate smooth delta sequence
        num_steps = max(int(duration * self.control_frequency), 2)

        # Use smooth interpolation weights (like minimum jerk)
        t = np.linspace(0, 1, num_steps)
        weights = 3 * t**2 - 2 * t**3  # Smooth S-curve

        delta_actions = []
        prev_weight = 0

        for weight in weights:
            # Calculate incremental delta for this step
            delta_weight = float(weight - prev_weight)

            delta_position = total_position_delta * delta_weight
            delta_euler = total_euler_delta * delta_weight

            delta_actions.append({
                'delta_position': delta_position,
                'delta_euler': delta_euler
            })

            prev_weight = weight

        # VERIFICATION: Check that delta actions sum to total displacement
        total_delta_pos = sum(action['delta_position'] for action in delta_actions)
        total_delta_euler = sum(action['delta_euler'] for action in delta_actions)

        pos_error = np.linalg.norm(total_delta_pos - total_position_delta)
        euler_error = np.linalg.norm(total_delta_euler - total_euler_delta)

        if pos_error > 1e-10:
            print(f"[WARN] Delta action position sum error: {pos_error:.12f}")
        if euler_error > 1e-10:
            print(f"[WARN] Delta action euler sum error: {euler_error:.12f}")

        print(f"Generated {len(delta_actions)} delta actions for {duration}s trajectory")
        print(f"Total position delta: {np.linalg.norm(total_position_delta):.6f}m")
        print(f"Total euler delta: {np.linalg.norm(total_euler_delta):.6f}rad")

        return delta_actions

    def _execute_delta_actions_without_recording(self, delta_actions, description):
        """
        Execute delta actions WITHOUT data recording (for Phase 0 initialization).
        Uses same delta accumulation logic but skips data collection calls.

        Args:
            delta_actions: List of delta action dictionaries
            description: Description for logging
        """
        print(f"Executing delta actions (no recording): {description}")

        if len(delta_actions) < 1:
            print("Warning: No delta actions to execute")
            return

        control_rate = rospy.Rate(self.control_frequency)
        init_time = rospy.Time.now().to_time()

        print(f"[INFO] Starting delta action execution with {len(delta_actions)} steps (no recording)...")

        for i, delta_action in enumerate(delta_actions):
            try:
                timestamp = rospy.Time.now().to_time() - init_time

                # Get delta actions for this step
                delta_xyz = delta_action['delta_position']
                delta_euler = delta_action['delta_euler']

                # Apply delta actions to current pose (same as recorded version)
                delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz')
                self.command_xyz += delta_xyz
                self.command_rotation = np.matmul(self.command_rotation, delta_rotation)

                # Create command transform
                command_transform = RigidTransform(
                    rotation=self.command_rotation,
                    translation=self.command_xyz,
                    from_frame='franka_tool',
                    to_frame='world'
                )

                # Send pose command using sensor publishing (NO data recording)
                pub_traj_gen_proto_msg = PosePositionSensorMessage(
                    id=i+1, timestamp=timestamp,
                    position=command_transform.translation, quaternion=command_transform.quaternion
                )
                fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                    id=i+1, timestamp=timestamp,
                    translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES,
                    rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
                )
                ros_pub_sensor_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        pub_traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                    feedback_controller_sensor_msg=sensor_proto2ros_msg(
                        fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
                )

                # Publish sensor values for real-time control
                self.robot.publish_sensor_values(ros_pub_sensor_msg)

                control_rate.sleep()

            except Exception as e:
                print(f"[WARN] Delta action step {i} failed: {e}")
                # FIXED: Recover like data_replayer.py - always re-establish baseline
                self._ee_pose_init()
                control_rate.sleep()
                continue

        print(f"[INFO] Delta action execution completed (no recording).")

        # VERIFICATION: Check final pose accuracy
        final_pose = self.robot.get_pose()
        command_pose = RigidTransform(
            rotation=self.command_rotation,
            translation=self.command_xyz,
            from_frame='franka_tool',
            to_frame='world'
        )

        pos_error = np.linalg.norm(final_pose.translation - command_pose.translation)
        quat_error = min(
            np.linalg.norm(final_pose.quaternion - command_pose.quaternion),
            np.linalg.norm(final_pose.quaternion + command_pose.quaternion)
        )

        print(f"[VERIFICATION] Final pose accuracy:")
        print(f"  Position error: {pos_error:.6f}m")
        print(f"  Orientation error: {quat_error:.6f}")

        if pos_error > 0.005:  # 5mm threshold
            print(f"[WARN] Large position error detected: {pos_error:.6f}m")
        if quat_error > 0.01:  # Small angle threshold
            print(f"[WARN] Large orientation error detected: {quat_error:.6f}")

    def _execute_delta_actions_with_recording(self, delta_actions, description):
        """
        Execute delta actions with data recording (exactly like original data collection).

        Args:
            delta_actions: List of delta action dictionaries
            description: Description for logging
        """
        print(f"Executing delta actions: {description}")

        if len(delta_actions) < 1:
            print("Warning: No delta actions to execute")
            return

        control_rate = rospy.Rate(self.control_frequency)
        init_time = rospy.Time.now().to_time()

        print(f"[INFO] Starting delta action execution with {len(delta_actions)} steps...")

        for i, delta_action in enumerate(delta_actions):
            try:
                timestamp = rospy.Time.now().to_time() - init_time

                # Get delta actions for this step
                delta_xyz = delta_action['delta_position']
                delta_euler = delta_action['delta_euler']

                # Apply delta actions to current pose (exactly like original data collection)
                delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz')
                self.command_xyz += delta_xyz
                self.command_rotation = np.matmul(self.command_rotation, delta_rotation)

                # Create command transform
                command_transform = RigidTransform(
                    rotation=self.command_rotation,
                    translation=self.command_xyz,
                    from_frame='franka_tool',
                    to_frame='world'
                )

                # Record data (exactly like original data collection)
                save_action = {
                    "delta": {
                        "position": delta_xyz,
                        "orientation": euler2quat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz'),
                        "euler_angle": delta_euler,
                    },
                    "abs": {
                        "position": copy.deepcopy(self.command_xyz),
                        "euler_angle": np.array([mat2euler(self.command_rotation, 'sxyz')])[0]
                    },
                    "gripper_width": self.robot.get_gripper_width() / FC.GRIPPER_WIDTH_MAX
                }

                # Update data collector
                self.data_collector.update_data_dict(
                    instruction="pull_drawer",
                    action=save_action,
                    timestamp=timestamp
                )

                # Send pose command using sensor publishing (exactly like original data collection)
                pub_traj_gen_proto_msg = PosePositionSensorMessage(
                    id=i+1, timestamp=timestamp,
                    position=command_transform.translation, quaternion=command_transform.quaternion
                )
                fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                    id=i+1, timestamp=timestamp,
                    translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES,
                    rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
                )
                ros_pub_sensor_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        pub_traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                    feedback_controller_sensor_msg=sensor_proto2ros_msg(
                        fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
                )

                # Publish sensor values for real-time control
                self.robot.publish_sensor_values(ros_pub_sensor_msg)

                control_rate.sleep()

            except Exception as e:
                print(f"[WARN] Delta action step {i} failed: {e}")
                # Recover like original data collection
                self._ee_pose_init()
                control_rate.sleep()
                continue

        print(f"[INFO] Delta action execution completed.")

        # VERIFICATION: Check final pose accuracy
        final_pose = self.robot.get_pose()
        command_pose = RigidTransform(
            rotation=self.command_rotation,
            translation=self.command_xyz,
            from_frame='franka_tool',
            to_frame='world'
        )

        pos_error = np.linalg.norm(final_pose.translation - command_pose.translation)
        quat_error = min(
            np.linalg.norm(final_pose.quaternion - command_pose.quaternion),
            np.linalg.norm(final_pose.quaternion + command_pose.quaternion)
        )

        print(f"[VERIFICATION] Final pose accuracy:")
        print(f"  Position error: {pos_error:.6f}m")
        print(f"  Orientation error: {quat_error:.6f}")

        if pos_error > 0.005:  # 5mm threshold
            print(f"[WARN] Large position error detected: {pos_error:.6f}m")
        if quat_error > 0.01:  # Small angle threshold
            print(f"[WARN] Large orientation error detected: {quat_error:.6f}")

    # REMOVED: _select_random_start_pose method - Phase 2 now starts from frame 0 like data_replayer

    def _execute_phase1_connection_to_first_frame(self, first_pose, duration=3.0):
        """
        Phase 1: Move from random pose to first frame of base episode (recorded).
        Simplified version that connects to frame 0 of base episode.

        Args:
            first_pose: First pose of base episode (frame 0)
            duration: Duration to move to starting pose
        """
        print("Phase 1: Moving to first frame of base episode (connection movement)...")

        # Initialize dynamic skill for phase 1 connection movement
        current_pose = self.robot.get_pose()
        self.robot.goto_pose(current_pose, duration=10, dynamic=True,
                           buffer_time=100000000, skill_desc='PHASE1_CONNECTION',
                           cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES,
                           ignore_virtual_walls=True)

        # Wait for dynamic skill to initialize
        time.sleep(FC.DYNAMIC_SKILL_WAIT_TIME)

        connection_delta_actions = self._generate_delta_action_sequence(first_pose, duration)
        self._execute_delta_actions_with_recording(connection_delta_actions, "Phase 1: Moving to base episode start")

        # Stop the connection movement skill
        self.robot.stop_skill()

        print("Phase 1 completed successfully")

    def _execute_phase2_base_episode(self, base_episode_data):
        """
        Phase 2: Execute complete base episode from frame 0 using data_replayer.py methodology.
        This exactly replicates the data replayer's proven approach for maximum accuracy.

        Args:
            base_episode_data: Base episode data dictionary
        """
        print("Phase 2: Executing complete base episode from frame 0 (data_replayer methodology)")

        # CRITICAL: Data replayer moves to HOME_POSE before establishing baseline
        # This ensures the starting pose is exactly correct for delta accumulation
        # The original base episodes were recorded starting from HOME_POSE baseline
        print("Phase 2: Moving to HOME_POSE like data_replayer.py robot_init()...")
        print("Phase 2: This ensures delta accumulation starts from the same baseline as original data")

        # Stop any existing dynamic skill first
        self.robot.stop_skill()

        # EXACT REPLICATION: Data replayer's robot_init() sequence
        # Move to HOME_POSE and start dynamic skill (exactly like data_replayer.py)
        self.robot.goto_pose(FC.HOME_POSE, duration=2, dynamic=True,
                           buffer_time=100000000, skill_desc='PHASE2_DATA_REPLAYER_INIT',
                           cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES,
                           ignore_virtual_walls=True)

        print("Phase 2: Robot moved to HOME_POSE, establishing baseline...")

        # EXACT REPLICATION: Data replayer calls ee_pose_init() immediately after robot_init()
        # This establishes baseline at HOME_POSE (current robot pose)
        self._ee_pose_init()
        print(f"Phase 2: Baseline established at HOME_POSE: {self.command_xyz}")

        # Execute the complete base episode using data replayer's exact logic
        self._replay_base_episode_like_data_replayer(base_episode_data)

        print("Phase 2 completed successfully")

    def _replay_base_episode_like_data_replayer(self, base_episode_data):
        """
        Replay base episode using EXACT data_replayer.py methodology.
        This is a direct adaptation of data_replayer.py's replay_action() method.
        """
        print("Replicating data_replayer.py's exact replay logic...")

        # EXACT COPY: Get sequences like data_replayer.py (always use delta mode)
        position_sequence = base_episode_data["action"]["end_effector"]["delta_position"]
        rotation_sequence = base_episode_data["action"]["end_effector"]["delta_euler"]
        gripper_width_sequence = base_episode_data["action"]["end_effector"]["gripper_width"]

        step = 0
        control_rate = rospy.Rate(self.control_frequency)
        print(f"[INFO] Starting replay data with {position_sequence.shape[0]} steps...")

        # EXACT COPY: Data replayer's time initialization
        self.init_time = rospy.Time.now().to_time()

        replay_step = position_sequence.shape[0]

        for i in range(replay_step):
            pos = position_sequence[i]
            rot = rotation_sequence[i]
            gripper_width = gripper_width_sequence[i]
            timestamp = rospy.Time.now().to_time() - self.init_time

            # EXACT COPY: Data replayer's delta action processing
            action = np.array([
                pos[0], pos[1], pos[2],
                rot[0], rot[1], rot[2],
                gripper_width
            ])

            delta_xyz, delta_euler, gripper = action[:3], action[3:6], action[-1]
            delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz')

            # EXACT COPY: Data replayer's pose accumulation
            self.command_xyz += delta_xyz
            self.command_rotation = np.matmul(self.command_rotation, delta_rotation)

            try:
                # EXACT COPY: Data replayer's command creation and publishing
                self.command_transform = RigidTransform(
                    rotation=self.command_rotation,
                    translation=self.command_xyz,
                    from_frame='franka_tool',
                    to_frame='world'
                )
                gripper_width_scaled = FC.GRIPPER_WIDTH_MAX * gripper

                # EXACT COPY: Data replayer's ROS message publishing
                pub_traj_gen_proto_msg = PosePositionSensorMessage(
                    id=step+1, timestamp=timestamp,
                    position=self.command_transform.translation,
                    quaternion=self.command_transform.quaternion
                )
                fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                    id=step+1, timestamp=timestamp,
                    translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES,
                    rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
                )
                ros_pub_sensor_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        pub_traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                    feedback_controller_sensor_msg=sensor_proto2ros_msg(
                        fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
                )

                self.robot.publish_sensor_values(ros_pub_sensor_msg)

                # EXACT COPY: Data replayer's gripper control
                current_gripper_width = self.robot.get_gripper_width()
                if abs(gripper_width_scaled - current_gripper_width) > 0.01:
                    grasp = True if gripper < 0.5 else False
                    self.robot.goto_gripper(gripper_width_scaled, grasp=grasp,
                                          force=FC.GRIPPER_MAX_FORCE/3.0, speed=0.12,
                                          block=False, skill_desc="control_gripper")

                # ADDITION: Record data (this is the key difference from data_replayer)
                save_action = {
                    "delta": {
                        "position": delta_xyz,
                        "orientation": euler2quat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz'),
                        "euler_angle": delta_euler,
                    },
                    "abs": {
                        "position": copy.deepcopy(self.command_xyz),
                        "euler_angle": np.array([mat2euler(self.command_rotation, 'sxyz')])[0]
                    },
                    "gripper_width": gripper
                }

                # Update data collector
                self.data_collector.update_data_dict(
                    instruction="pull_drawer",
                    action=save_action,
                    timestamp=timestamp
                )

            except Exception as e:
                # EXACT COPY: Data replayer's error recovery
                self._ee_pose_init()
                control_rate.sleep()
                print(f"[WARN] Move failed? : {e}")
                continue

            step += 1
            control_rate.sleep()

        self.robot.stop_skill()
        print("[INFO] Data replayer methodology replay completed.")

    def _execute_trajectory_with_recording(self, trajectory, description):
        """
        Execute a trajectory while recording data using dynamic control.

        Args:
            trajectory: List of RigidTransform poses
            description: Description for logging
        """
        print(f"Executing trajectory: {description}")

        if len(trajectory) < 2:
            print("Warning: Trajectory too short, skipping")
            return

        # Initialize dynamic control mode
        duration = len(trajectory) / self.control_frequency
        self.robot.goto_pose(trajectory[0], duration=duration, dynamic=True,
                           buffer_time=duration + 5, skill_desc=description,
                           cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES,
                           ignore_virtual_walls=True)

        # Wait for dynamic mode to initialize
        time.sleep(FC.DYNAMIC_SKILL_WAIT_TIME)

        control_rate = rospy.Rate(self.control_frequency)
        init_time = rospy.Time.now().to_time()

        for i, pose in enumerate(trajectory):
            try:
                timestamp = rospy.Time.now().to_time() - init_time

                # Calculate action (delta from previous pose)
                if i > 0:
                    prev_pose = trajectory[i-1]
                    delta_pos = pose.translation - prev_pose.translation
                    delta_rot_mat = pose.rotation @ prev_pose.rotation.T
                    delta_euler = mat2euler(delta_rot_mat, 'sxyz')
                else:
                    delta_pos = np.zeros(3)
                    delta_euler = np.zeros(3)

                # Get current gripper state
                gripper_width = self.robot.get_gripper_width()

                # Create action dictionary for data recording
                save_action = {
                    "delta": {
                        "position": delta_pos,
                        "orientation": euler2quat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz'),
                        "euler_angle": delta_euler,
                    },
                    "abs": {
                        "position": copy.deepcopy(pose.translation),
                        "euler_angle": np.array([mat2euler(pose.rotation, 'sxyz')])[0]
                    },
                    "gripper_width": gripper_width
                }

                # Update data collector
                self.data_collector.update_data_dict(
                    instruction="pull_drawer",
                    action=save_action,
                    timestamp=timestamp
                )

                # Send pose command to robot using sensor publishing (like original system)
                pub_traj_gen_proto_msg = PosePositionSensorMessage(
                    id=i+1, timestamp=timestamp,
                    position=pose.translation, quaternion=pose.quaternion
                )
                fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                    id=i+1, timestamp=timestamp,
                    translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES,
                    rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
                )
                ros_pub_sensor_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        pub_traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                    feedback_controller_sensor_msg=sensor_proto2ros_msg(
                        fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
                )

                # Publish sensor values for real-time control
                self.robot.publish_sensor_values(ros_pub_sensor_msg)

                control_rate.sleep()

            except Exception as e:
                print(f"Error executing trajectory step {i}: {e}")
                break

        # Stop the dynamic skill
        self.robot.stop_skill()

    # REMOVED: _execute_base_episode_sequence method - replaced with data_replayer methodology

    def robot_init_for_episode(self):
        """
        Initialize robot for new episode (exactly like data_collection.py).
        """
        print("Initializing robot for new episode...")
        # Exactly like data_collection.py main()
        self.robot.reset_joints()
        self.robot.open_gripper()

        # Start dynamic skill (exactly like data_collection.py)
        self.robot.goto_pose(FC.HOME_POSE, duration=10, dynamic=True,
                           buffer_time=100000000, skill_desc='MOVE',
                           cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES,
                           ignore_virtual_walls=True)

        # Initialize pose tracking from current position (exactly like data_collection.py)
        self._ee_pose_init()

        print("Robot initialized and ready for episode generation.")

    def generate_episode(self, base_episode_idx=0):
        """
        Generate a single new episode using the three-phase approach.

        Args:
            base_episode_idx: Index of base episode to use for Phase 2

        Returns:
            bool: Success status
        """
        try:
            print(f"\n=== Generating new episode using base episode {base_episode_idx} ===")

            # Stop the initialization skill before starting Phase 0
            self.robot.stop_skill()

            # Phase 0: Initialization movement (NOT recorded)
            random_pose = self._generate_random_pose()
            self._execute_phase0_initialization(random_pose, duration=5.0)

            # Get base episode data
            if base_episode_idx not in self.base_episodes:
                print(f"Error: Base episode {base_episode_idx} not found!")
                return False

            base_episode_data = self.base_episodes[base_episode_idx]

            # Get first pose from base episode for Phase 1 connection
            first_position = base_episode_data['state']['end_effector']['position'][0]
            first_orientation = base_episode_data['state']['end_effector']['orientation'][0]
            first_pose = RigidTransform(
                rotation=first_orientation,
                translation=first_position,
                from_frame='franka_tool',
                to_frame='world'
            )

            # Clear previous data and start recording (robot is now at random pose)
            self.data_collector.clear_data()
            print("[INFO] Data recording started from random pose")

            # Phase 1: Connection movement to first frame of base episode (recorded)
            self._execute_phase1_connection_to_first_frame(first_pose, duration=3.0)

            # Phase 2: Execute complete base episode from frame 0 using data_replayer methodology
            self._execute_phase2_base_episode(base_episode_data)

            print("Phase 2 completed. Stopping dynamic skill...")
            self.robot.stop_skill()

            # Save episode data immediately after Phase 2 ends
            episode_idx = self._get_next_episode_idx()
            success = self._save_episode(episode_idx)

            if success:
                print(f"✅ Episode {episode_idx} data saved successfully")
            else:
                print(f"❌ Failed to save episode {episode_idx}")
                return False

            return True

        except Exception as e:
            print(f"Error generating episode: {e}")
            # Make sure to stop any running skills
            try:
                self.robot.stop_skill()
            except:
                pass
            return False

    def _save_episode(self, episode_idx):
        """
        Save the collected episode data.

        Args:
            episode_idx: Episode index to save

        Returns:
            bool: Success status
        """
        try:
            episode_dir = os.path.join(self.new_dataset_dir, f"episode_{episode_idx}")
            os.makedirs(episode_dir, exist_ok=True)

            # Save data using VLADataCollector
            self.data_collector.save_data(episode_dir, episode_idx, is_compressed=False, is_save_video=True)

            # Save metadata
            metadata = {
                "task_name": "pull_drawer",
                "episode_idx": episode_idx,
                "action_steps": len(self.data_collector.data_dict["action"]["end_effector"]["delta_position"]),
                "instruction": "pull_drawer",
                "generation_method": "three_phase_trajectory",
                "phase0": "initialization_movement_not_recorded",
                "phase1": "connection_movement_recorded",
                "phase2": "base_episode_execution_recorded"
            }

            metadata_path = os.path.join(episode_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            print(f"Episode {episode_idx} saved successfully with {metadata['action_steps']} steps")
            return True

        except Exception as e:
            print(f"Error saving episode {episode_idx}: {e}")
            return False


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Three-phase trajectory data generation for pull_drawer dataset.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to generate.")
    parser.add_argument("--base_episode", type=int, default=0, help="Base episode index to use for Phase 2.")
    parser.add_argument("--base_dataset_dir", type=str, default="datasets/yukun/pull_drawer", help="Directory containing base dataset.")
    parser.add_argument("--new_dataset_dir", type=str, default="datasets/yukun/pull_drawer_new", help="Directory to save new episodes.")
    parser.add_argument("--random_base_episodes", action="store_true", help="Use random base episodes for each generation.")

    # Position offset ranges (in meters, per axis)
    parser.add_argument("--pos_x_range", type=float, nargs=2, default=[-0.05, 0.05], help="Position offset range for X axis [min, max] in meters (default: -0.05 0.05)")
    parser.add_argument("--pos_y_range", type=float, nargs=2, default=[-0.05, 0.05], help="Position offset range for Y axis [min, max] in meters (default: -0.05 0.05)")
    parser.add_argument("--pos_z_range", type=float, nargs=2, default=[-0.05, 0.05], help="Position offset range for Z axis [min, max] in meters (default: -0.05 0.05)")

    return parser.parse_args()


def main():
    """Main function to run the data generation."""
    args = get_arguments()

    print("=== Three-Phase Trajectory Data Generator ===")
    print(f"Base dataset: {args.base_dataset_dir}")
    print(f"New dataset: {args.new_dataset_dir}")
    print(f"Episodes to generate: {args.num_episodes}")

    try:
        # Initialize generator (FrankaArm will handle ROS node initialization)
        generator = ThreePhaseDataGenerator(
            args.base_dataset_dir,
            args.new_dataset_dir,
            pos_x_range=args.pos_x_range,
            pos_y_range=args.pos_y_range,
            pos_z_range=args.pos_z_range
        )

        print(f"Loaded {len(generator.base_episodes)} base episodes")
        print("Ready to start episode generation.")

        # Generate episodes with new flow: episode -> save -> user input -> robot init
        successful_episodes = 0

        # Initialize robot for first episode
        generator.robot_init_for_episode()

        for i in range(args.num_episodes):
            print(f"\n=== Episode {i+1}/{args.num_episodes} ===")

            # Ask user to start episode (robot is already at HOME_POSE)
            input("[INFO] Press enter to start episode generation")

            # Select base episode
            if args.random_base_episodes:
                base_episode_idx = np.random.choice(list(generator.base_episodes.keys()))
            else:
                base_episode_idx = args.base_episode

            print(f"Using base episode {base_episode_idx} for generation")

            # Generate episode (Phase 0 + Phase 1 + Phase 2 + Save)
            success = generator.generate_episode(base_episode_idx)

            if success:
                successful_episodes += 1
                print(f"✅ Episode {i+1} completed successfully")

                # Ask user for next episode (after data is saved, before moving to home)
                if i < args.num_episodes - 1:  # Not the last episode
                    input("[INFO] Press enter to continue to next episode")
                    # Initialize robot for next episode (move to HOME_POSE)
                    generator.robot_init_for_episode()
            else:
                print(f"❌ Failed to generate episode {i+1}")
                # Ask user if they want to continue
                if i < args.num_episodes - 1:  # Not the last episode
                    continue_choice = input("Continue with next episode? (y/n): ")
                    if continue_choice.lower() != 'y':
                        print("Episode generation stopped by user.")
                        break
                    # Initialize robot for next episode if continuing
                    generator.robot_init_for_episode()

        print(f"\n=== Generation Complete ===")
        print(f"Successfully generated {successful_episodes}/{args.num_episodes} episodes")

    except KeyboardInterrupt:
        print("\nData generation interrupted by user")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        print("Shutting down...")


if __name__ == "__main__":
    main()
