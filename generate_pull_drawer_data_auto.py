#!/usr/bin/env python3
"""
Enhanced trajectory generation script for pull_drawer dataset with automated push drawer reset.

Phase 0: Initialization movement (NOT recorded) - move from HOME_POSE to random pose
Phase 1: Connection movement (recorded) - move from random pose to pull episode start
Phase 2: Execute pull drawer episode (recorded)
Phase 3: Connection movement (NOT recorded) - move from pull end to push episode start frame
Phase 4: Execute push drawer episode from start frame (NOT recorded) - automated reset

Only Phase 1 and Phase 2 are recorded for training data. Phase 3 and 4 provide automated reset.

Usage:
    python3 generate_pull_drawer_data_auto.py --num_episodes 5 --base_episode 2 --push_episode 0 --push_start_frame 10 --new_dataset_dir datasets/yukun/pull_drawer_new2 --pos_x_range -0.10 0.10 --pos_y_range -0.17 0.0 --pos_z_range -0.05 0.10
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import copy
import threading
# Removed scipy dependency - using numpy interpolation instead
from transforms3d.euler import euler2quat, euler2mat, mat2euler
from pynput.keyboard import Listener

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
                 push_dataset_dir="push_drawer", push_start_frame=10, pos_x_range=None, pos_y_range=None, pos_z_range=None, debug=False):
        """
        Initialize the three-phase data generator with push drawer reset functionality.

        Args:
            base_dataset_dir: Directory containing the original pull_drawer dataset
            new_dataset_dir: Directory to save new generated episodes
            push_dataset_dir: Directory containing the push_drawer dataset for reset
            push_start_frame: Frame number to start push episode execution from (default: 10)
            pos_x_range: Position offset range for X axis [min, max] in meters
            pos_y_range: Position offset range for Y axis [min, max] in meters
            pos_z_range: Position offset range for Z axis [min, max] in meters
            debug: Enable debug logging
        """
        self.base_dataset_dir = base_dataset_dir
        self.new_dataset_dir = new_dataset_dir
        self.push_dataset_dir = push_dataset_dir
        self.push_start_frame = push_start_frame
        self.debug = debug
        self.control_frequency = 5  # Hz, same as original data collection

        # Initialize robot and cameras
        # FrankaArm will handle ROS node initialization automatically
        self.robot = FrankaArm()
        self.cameras = RealsenseAPI()
        self.data_collector = VLADataCollector(self.robot, self.cameras)

        # Load base datasets
        self.base_episodes = self._load_base_episodes()
        self.push_episodes = self._load_push_episodes()

        # Create new dataset directory
        os.makedirs(self.new_dataset_dir, exist_ok=True)

        # Random pose delta limits (relative to current position)
        # Position deltas: configurable per axis
        # Orientation deltas: -10 to 10 degrees in each axis
        deg2rad = np.pi / 180  # Easy to change: modify degrees above, conversion happens here
        self.random_delta_limits = {
            'position_x': pos_x_range if pos_x_range is not None else [-0.10, 0.10],
            'position_y': pos_y_range if pos_y_range is not None else [-0.20, 0.0],
            'position_z': pos_z_range if pos_z_range is not None else [-0.05, 0.10],
            'orientation': [-15 * deg2rad, 15 * deg2rad]  # ±10 degrees in each axis
        }

        # Initialize keyboard listener for 'q' key detection (same as data collection)
        self.quit_signal = False
        self.listener = Listener(on_release=self.on_key_release)
        self.listener.start()

    def on_key_release(self, key):
        """
        Key handler for key releases (same as data collection).
        Args:
            key: key that was released
        """
        try:
            if key.char == "q":
                self.quit_signal = True
                print("[INFO] 'q' key detected - aborting episode...")
        except AttributeError:
            pass

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

    def _load_push_episodes(self):
        """Load all push episodes from the push_drawer dataset."""
        push_episodes = {}
        if not os.path.exists(self.push_dataset_dir):
            print(f"Warning: Push dataset directory {self.push_dataset_dir} not found. Push reset will be disabled.")
            return push_episodes

        episode_dirs = [d for d in os.listdir(self.push_dataset_dir)
                       if d.startswith('episode_') and os.path.isdir(os.path.join(self.push_dataset_dir, d))]

        for episode_dir in episode_dirs:
            episode_idx = int(episode_dir.split('_')[1])
            data_path = os.path.join(self.push_dataset_dir, episode_dir, 'data.npy')

            if os.path.exists(data_path):
                try:
                    episode_data = np.load(data_path, allow_pickle=True).item()
                    push_episodes[episode_idx] = episode_data
                    print(f"Loaded push episode {episode_idx}")
                except Exception as e:
                    print(f"Failed to load push episode {episode_idx}: {e}")

        return push_episodes

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

    def _generate_complete_trajectory_sequences(self, base_episode_data, random_pose, push_episode_data=None):
        """
        Generate complete trajectory sequences for all phases BEFORE execution.
        Now includes optional push drawer reset phases.
        Returns sequences in data replayer format.

        Args:
            base_episode_data: Base episode data for Phase 2 (pull drawer)
            random_pose: Pre-generated random pose for Phase 0 target
            push_episode_data: Optional push episode data for reset phases
        """
        print("Generating complete trajectory sequences for all phases...")

        # Phase 0: HOME_POSE to random pose (not recorded)
        phase0_positions, phase0_eulers, phase0_grippers = self._generate_phase0_sequence(random_pose)

        # Phase 1: Random pose to base episode start (recorded)
        phase1_positions, phase1_eulers, phase1_grippers = self._generate_phase1_sequence(base_episode_data, random_pose)

        # Phase 2: Complete base episode (recorded)
        phase2_positions = base_episode_data["action"]["end_effector"]["delta_position"]
        phase2_eulers = base_episode_data["action"]["end_effector"]["delta_euler"]
        phase2_grippers = base_episode_data["action"]["end_effector"]["gripper_width"]

        # Store phase boundaries for data recording (only Phase 1 and 2 are recorded)
        self.phase0_end = len(phase0_positions)
        self.phase1_end = self.phase0_end + len(phase1_positions)
        self.phase2_end = self.phase1_end + len(phase2_positions)

        # Initialize sequences with recorded phases
        complete_positions = np.vstack([phase0_positions, phase1_positions, phase2_positions])
        complete_eulers = np.vstack([phase0_eulers, phase1_eulers, phase2_eulers])
        complete_grippers = np.concatenate([phase0_grippers, phase1_grippers, phase2_grippers])

        # Add push drawer reset phases if available (not recorded)
        if push_episode_data is not None:
            # Phase 3: Connection to push episode frame 10 (not recorded)
            phase3_positions, phase3_eulers, phase3_grippers = self._generate_push_connection_sequence(
                base_episode_data, push_episode_data)

            # Phase 4: Execute push episode from frame 10 (not recorded)
            phase4_positions, phase4_eulers, phase4_grippers = self._generate_push_execution_sequence(push_episode_data)

            # Add push phases to complete trajectory
            complete_positions = np.vstack([complete_positions, phase3_positions, phase4_positions])
            complete_eulers = np.vstack([complete_eulers, phase3_eulers, phase4_eulers])
            complete_grippers = np.concatenate([complete_grippers, phase3_grippers, phase4_grippers])

            # Calculate steps to be saved (Phase 1 + Phase 2)
            steps_to_save = len(phase1_positions) + len(phase2_positions)
            print(f"Generated trajectory: Phase0={len(phase0_positions)}, Phase1={len(phase1_positions)}, Phase2={len(phase2_positions)}, Phase3={len(phase3_positions)}, Phase4={len(phase4_positions)} steps")
            print(f"📊 EXPECTED STEPS TO SAVE: {steps_to_save} (Phase1: {len(phase1_positions)} + Phase2: {len(phase2_positions)})")
            print(f"Phase 0: Current robot pose → Random pose (not recorded)")
            print(f"Phase 1: Random pose → Base episode start (recorded)")
            print(f"Phase 2: Execute pull drawer episode (recorded)")
            print(f"Phase 3: Connection to push episode frame 10 (not recorded)")
            print(f"Phase 4: Execute push drawer episode from frame 10 (not recorded)")
        else:
            # Calculate steps to be saved (Phase 1 + Phase 2)
            steps_to_save = len(phase1_positions) + len(phase2_positions)
            print(f"Generated trajectory: Phase0={len(phase0_positions)}, Phase1={len(phase1_positions)}, Phase2={len(phase2_positions)} steps")
            print(f"📊 EXPECTED STEPS TO SAVE: {steps_to_save} (Phase1: {len(phase1_positions)} + Phase2: {len(phase2_positions)})")
            print(f"Phase 0: Current robot pose → Random pose (not recorded)")
            print(f"Phase 1: Random pose → Base episode start (recorded)")
            print(f"Phase 2: Execute pull drawer episode (recorded)")
            print("[INFO] No push episode data provided - skipping reset phases")

        return complete_positions, complete_eulers, complete_grippers

    def _generate_phase0_sequence(self, random_pose):
        """Generate Phase 0: Current robot pose to random pose (initialization, not recorded)"""
        # Use CURRENT robot pose as start (not HOME_POSE)
        start_pose = RigidTransform(
            rotation=self.init_rotation,
            translation=self.init_xyz,
            from_frame='franka_tool',
            to_frame='world'
        )
        target_pose = random_pose
        duration = 1.5
        num_steps = int(duration * self.control_frequency)

        # Simple linear interpolation for Phase 0
        positions = self._interpolate_positions([start_pose.translation, target_pose.translation], num_steps)
        orientations = self._interpolate_orientations([start_pose.quaternion, target_pose.quaternion], num_steps)
        grippers = np.full(num_steps, 1.0)  # Open gripper

        # Convert to delta sequences
        delta_positions = np.diff(positions, axis=0, prepend=positions[0:1])
        delta_eulers = self._quaternions_to_delta_eulers(orientations)

        return delta_positions, delta_eulers, grippers

    def _generate_phase1_sequence(self, base_episode_data, random_pose):
        """Generate Phase 1: Random pose to base episode start (recorded)"""
        start_pose = random_pose  # Start from the random pose reached in Phase 0

        # Get first pose from base episode
        first_position = base_episode_data['state']['end_effector']['position'][0]
        first_orientation = base_episode_data['state']['end_effector']['orientation'][0]
        end_pose = RigidTransform(rotation=first_orientation, translation=first_position,
                                from_frame='franka_tool', to_frame='world')

        duration = np.random.uniform(2.2, 3.0)
        num_steps = int(duration * self.control_frequency)

        # Smooth interpolation for Phase 1
        positions = self._interpolate_positions([start_pose.translation, end_pose.translation], num_steps)
        orientations = self._interpolate_orientations([start_pose.quaternion, end_pose.quaternion], num_steps)
        grippers = np.full(num_steps, 1.0)  # Open gripper

        # Convert to delta sequences
        delta_positions = np.diff(positions, axis=0, prepend=positions[0:1])
        delta_eulers = self._quaternions_to_delta_eulers(orientations)

        return delta_positions, delta_eulers, grippers

    def _generate_push_connection_sequence(self, pull_episode_data, push_episode_data):
        """Generate Phase 3: Connection from end of pull episode to specified frame of push episode (not recorded)"""
        # Start pose: end of pull episode (last frame)
        pull_end_position = pull_episode_data['state']['end_effector']['position'][-1]
        pull_end_orientation = pull_episode_data['state']['end_effector']['orientation'][-1]
        start_pose = RigidTransform(rotation=pull_end_orientation, translation=pull_end_position,
                                  from_frame='franka_tool', to_frame='world')

        # Target pose: specified frame of push episode
        push_start_position = push_episode_data['state']['end_effector']['position'][self.push_start_frame]
        push_start_orientation = push_episode_data['state']['end_effector']['orientation'][self.push_start_frame]
        target_pose = RigidTransform(rotation=push_start_orientation, translation=push_start_position,
                                   from_frame='franka_tool', to_frame='world')

        duration = 1.5  # Connection duration
        num_steps = int(duration * self.control_frequency)

        # Smooth interpolation for connection
        positions = self._interpolate_positions([start_pose.translation, target_pose.translation], num_steps)
        orientations = self._interpolate_orientations([start_pose.quaternion, target_pose.quaternion], num_steps)
        grippers = np.full(num_steps, 1.0)  # Open gripper

        # Convert to delta sequences
        delta_positions = np.diff(positions, axis=0, prepend=positions[0:1])
        delta_eulers = self._quaternions_to_delta_eulers(orientations)

        return delta_positions, delta_eulers, grippers

    def _generate_push_execution_sequence(self, push_episode_data):
        """Generate Phase 4: Execute push episode starting from specified frame (not recorded)"""
        # Extract push episode data starting from specified frame
        push_positions = push_episode_data["action"]["end_effector"]["delta_position"][self.push_start_frame:]
        push_eulers = push_episode_data["action"]["end_effector"]["delta_euler"][self.push_start_frame:]
        push_grippers = push_episode_data["action"]["end_effector"]["gripper_width"][self.push_start_frame:]

        return push_positions, push_eulers, push_grippers

    def _interpolate_positions(self, waypoints, num_points):
        """Interpolate positions using numpy linear interpolation"""
        if len(waypoints) < 2:
            return np.array(waypoints)

        waypoints = np.array(waypoints)
        t_waypoints = np.linspace(0, 1, len(waypoints))
        t_interp = np.linspace(0, 1, num_points)

        # Use numpy linear interpolation for each axis
        interp_x = np.interp(t_interp, t_waypoints, waypoints[:, 0])
        interp_y = np.interp(t_interp, t_waypoints, waypoints[:, 1])
        interp_z = np.interp(t_interp, t_waypoints, waypoints[:, 2])

        return np.column_stack([interp_x, interp_y, interp_z])

    def _interpolate_orientations(self, quaternions, num_points):
        """Interpolate orientations using proper SLERP"""
        if len(quaternions) < 2:
            return np.array(quaternions)

        t_waypoints = np.linspace(0, 1, len(quaternions))
        t_interp = np.linspace(0, 1, num_points)

        result = []
        for t in t_interp:
            # Find surrounding waypoints
            idx = np.searchsorted(t_waypoints, t) - 1
            idx = max(0, min(idx, len(quaternions) - 2))

            # Local interpolation parameter
            if t_waypoints[idx + 1] == t_waypoints[idx]:
                local_t = 0
            else:
                local_t = (t - t_waypoints[idx]) / (t_waypoints[idx + 1] - t_waypoints[idx])

            # SLERP between adjacent quaternions
            q_interp = self._slerp(quaternions[idx], quaternions[idx + 1], local_t)
            result.append(q_interp)

        return np.array(result)

    def _slerp(self, q1, q2, t):
        """Proper spherical linear interpolation (SLERP) for quaternions"""
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

    def _execute_complete_trajectory(self, position_sequence, rotation_sequence, gripper_sequence):
        """
        Execute complete trajectory using data replayer methodology.
        Single baseline establishment, unified execution loop.
        Returns True if completed successfully, False if aborted by 'q' key.
        """
        print(f"Executing complete trajectory with {len(position_sequence)} steps...")

        # CRITICAL: Single baseline establishment (already done before trajectory generation)
        # Baseline was established using current robot pose before trajectory generation
        self.command_xyz = self.init_xyz.copy()
        self.command_rotation = self.init_rotation.copy()
        print(f"Using established baseline at: {self.command_xyz}")

        # Initialize execution
        step = 0
        control_rate = rospy.Rate(self.control_frequency)
        self.init_time = rospy.Time.now().to_time()

        # Start data recording from Phase 1
        self.data_collector.clear_data()
        print("[INFO] Data recording will start from Phase 1")

        for i in range(len(position_sequence)):
            # Check for 'q' key abort signal (same as data collection)
            if self.quit_signal:
                print("[INFO] Episode aborted by user ('q' key pressed)")
                self.robot.stop_skill()
                return False

            try:
                timestamp = rospy.Time.now().to_time() - self.init_time

                # Get delta actions for this step (data replayer format)
                delta_xyz = position_sequence[i]
                delta_euler = rotation_sequence[i]
                gripper_width = gripper_sequence[i]

                # EXACT COPY: Data replayer's pose accumulation
                delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz')
                self.command_xyz += delta_xyz
                self.command_rotation = np.matmul(self.command_rotation, delta_rotation)

                # EXACT COPY: Data replayer's command creation
                self.command_transform = RigidTransform(
                    rotation=self.command_rotation,
                    translation=self.command_xyz,
                    from_frame='franka_tool',
                    to_frame='world'
                )
                gripper_width_scaled = FC.GRIPPER_WIDTH_MAX * gripper_width

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
                    grasp = True if gripper_width < 0.5 else False
                    self.robot.goto_gripper(gripper_width_scaled, grasp=grasp,
                                          force=FC.GRIPPER_MAX_FORCE/3.0, speed=0.12,
                                          block=False, skill_desc="control_gripper")

                # Record data only from Phase 1 to end of Phase 2 (pull drawer execution)
                # Phase 3 and 4 (push drawer reset) are NOT recorded
                if i >= self.phase0_end and i < self.phase2_end:
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
                        "gripper_width": gripper_width
                    }

                    self.data_collector.update_data_dict(
                        instruction="pull_drawer",
                        action=save_action,
                        timestamp=timestamp
                    )

                # DEBUG: Log recording boundaries and current step
                if self.debug:
                    if i == 0:
                        print(f"[DEBUG] Recording boundaries: phase0_end={self.phase0_end}, phase2_end={self.phase2_end}, total_steps={len(position_sequence)}")
                    if i == self.phase0_end:
                        print(f"[DEBUG] Step {i}: Starting data recording (Phase 1)")
                    if i == self.phase2_end:
                        print(f"[DEBUG] Step {i}: Stopping data recording (end of Phase 2)")
                    if i == len(position_sequence) - 1:
                        print(f"[DEBUG] Step {i}: Final trajectory step")

            except Exception as e:
                # EXACT COPY: Data replayer's error recovery
                self._ee_pose_init()
                control_rate.sleep()
                print(f"[WARN] Move failed? : {e}")
                continue

            step += 1
            control_rate.sleep()

        # DEBUG: Log final robot state before stopping skill
        if self.debug:
            final_pose = self.robot.get_pose()
            print(f"[DEBUG] Final robot pose before stop: {final_pose.translation}")
            print(f"[DEBUG] Recorded data steps: {len(self.data_collector.data_dict['action']['end_effector']['delta_position'])}")

        # DO NOT call robot.stop_skill() here - it will be called by _stop_skill_safely()
        print("[INFO] Complete trajectory execution finished.")
        return True

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

    # REMOVED: _execute_phase0_initialization - using unified trajectory system

    # REMOVED: All old phase-specific execution methods
    # REMOVED: _execute_delta_actions_without_recording
    # REMOVED: _execute_delta_actions_with_recording
    # REMOVED: _execute_phase1_connection_to_first_frame
    # REMOVED: _execute_phase2_base_episode
    # REMOVED: _replay_base_episode_like_data_replayer
    # REMOVED: _execute_trajectory_with_recording

        pass  # All old methods removed - using unified trajectory system

    def _ensure_action_client_ready(self):
        """
        Simplified check to ensure robot is ready for new episode.
        """
        import time

        print("[INFO] Ensuring robot is ready for new episode...")

        # Simple check using franka interface status (more reliable than action client state)
        max_wait_time = 3.0  # seconds - increased timeout
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                franka_status = self.robot._get_current_franka_interface_status().franka_interface_status
                if franka_status.is_ready and not self.robot._in_skill:
                    print("[INFO] Robot confirmed ready for new episode")
                    # Additional small delay to ensure everything is settled
                    time.sleep(0.2)
                    return
            except Exception as e:
                print(f"[WARN] Error checking robot status: {e}")

            time.sleep(0.1)

        print("[WARN] Robot readiness timeout, but proceeding...")
        # Force a longer delay if timeout occurred
        time.sleep(1.0)

    def _stop_skill_safely(self):
        """
        Safely stop the current skill without triggering PREEMPTING/DONE race condition.
        This replaces the problematic robot.stop_skill() method.
        """
        import time

        print("[INFO] Safely stopping skill...")

        # DEBUG: Log current robot and action client state
        if self.debug:
            print(f"[DEBUG] Robot connected: {self.robot._connected}")
            print(f"[DEBUG] Robot in skill: {self.robot._in_skill}")
            try:
                client_state = self.robot._client.get_state()
                print(f"[DEBUG] Action client state: {client_state}")
            except Exception as e:
                print(f"[DEBUG] Could not get action client state: {e}")

        if not self.robot._connected:
            print("[INFO] Robot not connected")
            return

        # Check action client state before attempting to stop
        try:
            client_state = self.robot._client.get_state()
            if client_state == 3:  # DONE state
                print("[INFO] Skill already completed successfully")
                self.robot._in_skill = False
                return
            elif client_state == 4:  # PREEMPTED state
                print("[INFO] Skill already preempted")
                self.robot._in_skill = False
                return
            elif not self.robot._in_skill:
                print("[INFO] No active skill to stop")
                return
        except Exception as e:
            print(f"[DEBUG] Could not check action client state: {e}")
            if not self.robot._in_skill:
                print("[INFO] No active skill to stop")
                return

        # Cancel the goal
        print("[INFO] Cancelling current goal...")
        try:
            self.robot._client.cancel_goal()
            if self.debug:
                print("[DEBUG] Goal cancellation sent successfully")
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Error during goal cancellation: {e}")

        # Wait for cancellation to complete WITHOUT using wait_for_result() polling
        # This avoids the PREEMPTING/DONE race condition
        max_wait_time = 3.0  # seconds
        start_time = time.time()

        print("[INFO] Waiting for skill cancellation to complete...")
        while time.time() - start_time < max_wait_time:
            # Check franka interface status instead of action client state
            try:
                franka_status = self.robot._get_current_franka_interface_status().franka_interface_status
                if franka_status.is_ready:
                    # Franka interface is ready, skill should be stopped
                    self.robot._in_skill = False
                    print("[INFO] Skill safely stopped (franka interface ready)")
                    return
            except Exception as e:
                print(f"[WARN] Error checking franka status: {e}")

            time.sleep(0.05)  # 50ms sleep to avoid tight polling

        # Timeout reached, force reset the skill state
        print("[WARN] Skill stop timeout, forcing state reset...")
        self.robot._in_skill = False

        # Additional verification: ensure robot can accept new commands
        print("[INFO] Verifying robot is ready for new commands...")
        max_verify_time = 2.0
        verify_start = time.time()

        while time.time() - verify_start < max_verify_time:
            try:
                # Use franka interface status instead of is_skill_done to avoid action client issues
                franka_status = self.robot._get_current_franka_interface_status().franka_interface_status
                if franka_status.is_ready and not self.robot._in_skill:
                    print("[INFO] Robot confirmed ready for new commands")
                    return
            except Exception as e:
                print(f"[WARN] Error verifying robot readiness: {e}")
            time.sleep(0.1)

        print("[WARN] Robot readiness verification timeout, but proceeding...")

    def robot_init_for_episode(self):
        """
        Initialize robot for new episode (exactly like data_collection.py).
        """
        print("Initializing robot for new episode...")

        # DEBUG: Log current robot pose before initialization
        if self.debug:
            try:
                current_pose = self.robot.get_pose()
                print(f"[DEBUG] Robot pose before initialization: {current_pose.translation}")
            except Exception as e:
                print(f"[DEBUG] Could not get robot pose before init: {e}")

        # CRITICAL FIX: Ensure action client is in clean state before starting new episode
        self._ensure_action_client_ready()

        # Additional small delay to ensure robot is fully settled after previous episode
        import time
        time.sleep(0.5)
        print("Starting robot reset sequence...")

        # Exactly like data_collection.py main()
        print("Resetting joints...")

        # Final safety check before reset_joints to avoid "Cannot send another command" error
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Check if we can send commands using franka interface status
                franka_status = self.robot._get_current_franka_interface_status().franka_interface_status
                if not franka_status.is_ready:
                    print(f"[WARN] Franka interface not ready on attempt {attempt+1}, waiting...")
                    time.sleep(1.0)
                    continue

                if self.robot._in_skill:
                    print(f"[WARN] Robot still in skill on attempt {attempt+1}, waiting...")
                    time.sleep(1.0)
                    continue

                # Attempt reset_joints
                self.robot.reset_joints()
                print("Joint reset completed successfully")
                break

            except ValueError as e:
                if "Cannot send another command" in str(e):
                    print(f"[WARN] Command conflict on attempt {attempt+1}: {e}")
                    if attempt < max_attempts - 1:
                        print("Waiting before retry...")
                        time.sleep(2.0)
                        continue
                    else:
                        print("[ERROR] Failed to reset joints after all attempts")
                        raise
                else:
                    raise
            except Exception as e:
                print(f"[ERROR] Unexpected error during joint reset: {e}")
                raise

        print("Opening gripper...")
        self.robot.open_gripper()

        # Start dynamic skill (exactly like data_collection.py)
        print("Moving to HOME_POSE...")
        self.robot.goto_pose(FC.HOME_POSE, duration=10, dynamic=True,
                           buffer_time=100000000, skill_desc='MOVE',
                           cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES,
                           ignore_virtual_walls=True)
        print("HOME_POSE movement completed.")

        # Initialize pose tracking from current position (exactly like data_collection.py)
        self._ee_pose_init()

        # DEBUG: Verify robot is actually at HOME_POSE
        if self.debug:
            try:
                current_pose = self.robot.get_pose()
                home_pose = FC.HOME_POSE
                position_diff = np.linalg.norm(current_pose.translation - home_pose.translation)
                print(f"[DEBUG] Robot pose after HOME_POSE: {current_pose.translation}")
                print(f"[DEBUG] Expected HOME_POSE: {home_pose.translation}")
                print(f"[DEBUG] Position difference: {position_diff:.4f}m")
                if position_diff > 0.05:  # 5cm tolerance
                    print(f"[WARNING] Robot may not be at HOME_POSE! Difference: {position_diff:.4f}m")
            except Exception as e:
                print(f"[DEBUG] Could not verify HOME_POSE: {e}")

        print("Robot initialized and ready for episode generation.")

    def generate_episode(self, base_episode_idx=0, push_episode_idx=None):
        """
        Generate a single new episode using the enhanced approach with push drawer reset.
        Handles 'q' key abort and redo functionality.

        Args:
            base_episode_idx: Index of base episode to use for Phase 2 (pull drawer)
            push_episode_idx: Index of push episode to use for reset phases (optional)

        Returns:
            bool: Success status
        """
        while True:  # Loop for redo functionality
            try:
                print(f"\n=== Generating new episode using base episode {base_episode_idx} ===")

                # Reset quit signal for this episode attempt
                self.quit_signal = False

                # Stop the initialization skill before starting Phase 0
                self.robot.stop_skill()

                # Robot should now be at current pose, establish baseline for random pose generation
                current_pose = self.robot.get_pose()
                self.init_xyz = current_pose.translation
                self.init_rotation = current_pose.rotation

                # Generate random pose for Phase 0 target (using CURRENT robot pose as baseline)
                random_pose = self._generate_random_pose()
                print(f"Generated random target pose for Phase 0")
                print(f"Current robot pose: {self.init_xyz}")
                print(f"Random target pose: {random_pose.translation}")
                print(f"Position offset: {random_pose.translation - self.init_xyz}")

                # Get base episode data
                if base_episode_idx not in self.base_episodes:
                    print(f"Error: Base episode {base_episode_idx} not found!")
                    return False

                base_episode_data = self.base_episodes[base_episode_idx]

                # Get push episode data if available and requested
                push_episode_data = None
                if push_episode_idx is not None and push_episode_idx in self.push_episodes:
                    push_episode_data = self.push_episodes[push_episode_idx]
                    print(f"Using push episode {push_episode_idx} for reset phases")
                elif len(self.push_episodes) > 0:
                    # Use first available push episode if none specified
                    push_episode_idx = list(self.push_episodes.keys())[0]
                    push_episode_data = self.push_episodes[push_episode_idx]
                    print(f"Using default push episode {push_episode_idx} for reset phases")
                else:
                    print("[INFO] No push episodes available - skipping reset phases")

                # NEW UNIFIED SYSTEM: Generate complete trajectory sequences for all phases
                position_sequence, rotation_sequence, gripper_sequence = self._generate_complete_trajectory_sequences(
                    base_episode_data, random_pose, push_episode_data)

                # Initialize dynamic skill for entire trajectory execution
                current_pose = self.robot.get_pose()
                self.robot.goto_pose(current_pose, duration=10, dynamic=True,
                                   buffer_time=100000000, skill_desc='UNIFIED_TRAJECTORY_EXECUTION',
                                   cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES,
                                   ignore_virtual_walls=True)

                # Wait for dynamic skill to initialize
                time.sleep(FC.DYNAMIC_SKILL_WAIT_TIME)

                # Execute complete trajectory using data replayer methodology
                trajectory_success = self._execute_complete_trajectory(position_sequence, rotation_sequence, gripper_sequence)

                print("Trajectory execution finished. Stopping dynamic skill...")

                # DEBUG: Log robot state before stopping
                if self.debug:
                    try:
                        current_pose = self.robot.get_pose()
                        print(f"[DEBUG] Robot pose before skill stop: {current_pose.translation}")
                    except Exception as e:
                        print(f"[DEBUG] Could not get robot pose: {e}")

                # CRITICAL FIX: Use custom stop method to avoid PREEMPTING/DONE race condition
                self._stop_skill_safely()

                # DEBUG: Log robot state after stopping
                if self.debug:
                    try:
                        current_pose = self.robot.get_pose()
                        print(f"[DEBUG] Robot pose after skill stop: {current_pose.translation}")
                    except Exception as e:
                        print(f"[DEBUG] Could not get robot pose after stop: {e}")

                # Handle abort/redo logic (same as data collection)
                if not trajectory_success:  # Episode was aborted by 'q' key
                    print("[INFO] Episode aborted. Waiting 2.5s before redo...")
                    time.sleep(2.5)
                    print("[INFO] Redoing episode...")
                    # Reset robot to home pose for redo
                    self.robot_init_for_episode()
                    continue  # Redo the episode

                # Episode completed successfully - save data
                episode_idx = self._get_next_episode_idx()

                # DEBUG: Check data collector state before saving
                recorded_steps = len(self.data_collector.data_dict['action']['end_effector']['delta_position'])
                print(f"📊 ACTUAL STEPS RECORDED: {recorded_steps}")
                if self.debug:
                    print(f"[DEBUG] Data collector steps before save: {recorded_steps}")

                success = self._save_episode(episode_idx)

                if success:
                    print(f"✅ Episode {episode_idx} data saved successfully")
                    return True
                else:
                    print(f"❌ Failed to save episode {episode_idx}")
                    return False

            except Exception as e:
                print(f"Error generating episode: {e}")
                # Make sure to stop any running skills
                try:
                    self._stop_skill_safely()
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

            # DEBUG: Check data collector state right before save
            if self.debug:
                steps_before_save = len(self.data_collector.data_dict['action']['end_effector']['delta_position'])
                print(f"[DEBUG] Steps in data_collector right before save: {steps_before_save}")

            # Save data using VLADataCollector
            self.data_collector.save_data(episode_dir, episode_idx, is_compressed=False, is_save_video=True)

            # DEBUG: Check what was actually saved
            if self.debug:
                steps_after_save = len(self.data_collector.data_dict['action']['end_effector']['delta_position'])
                print(f"[DEBUG] Steps in data_collector after save: {steps_after_save}")

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

    def cleanup(self):
        """Clean up resources including keyboard listener."""
        try:
            if hasattr(self, 'listener') and self.listener:
                self.listener.stop()
                print("[INFO] Keyboard listener stopped")
        except Exception as e:
            print(f"[WARN] Error stopping keyboard listener: {e}")


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced trajectory data generation for pull_drawer dataset with push drawer reset.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to generate.")
    parser.add_argument("--base_episode", type=int, default=0, help="Base episode index to use for Phase 2 (pull drawer).")
    parser.add_argument("--push_episode", type=int, default=0, help="Push episode index to use for reset phases (optional).")
    parser.add_argument("--push_start_frame", type=int, default=15, help="Frame number to start push episode execution from (default: 10).")
    parser.add_argument("--base_dataset_dir", type=str, default="datasets/yukun/pull_drawer", help="Directory containing base pull_drawer dataset.")
    parser.add_argument("--push_dataset_dir", type=str, default="datasets/yukun/push_drawer", help="Directory containing push_drawer dataset.")
    parser.add_argument("--new_dataset_dir", type=str, default="datasets/yukun/pull_drawer_new", help="Directory to save new episodes.")
    parser.add_argument("--random_base_episodes", action="store_true", help="Use random base episodes for each generation.")
    parser.add_argument("--random_push_episodes", action="store_true", help="Use random push episodes for each reset.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    # Position offset ranges (in meters, per axis)
    parser.add_argument("--pos_x_range", type=float, nargs=2, default=[-0.1, 0.1], help="Position offset range for X axis [min, max] in meters (default: -0.05 0.05)")
    parser.add_argument("--pos_y_range", type=float, nargs=2, default=[-0.30, 0.0], help="Position offset range for Y axis [min, max] in meters (default: -0.05 0.05)")
    parser.add_argument("--pos_z_range", type=float, nargs=2, default=[-0.05, 0.05], help="Position offset range for Z axis [min, max] in meters (default: -0.05 0.05)")

    return parser.parse_args()


def main():
    """Main function to run the data generation."""
    args = get_arguments()

    print("=== Enhanced Pull-Drawer Data Generator with Push Reset ===")
    print(f"Base dataset: {args.base_dataset_dir}")
    print(f"Push dataset: {args.push_dataset_dir}")
    print(f"New dataset: {args.new_dataset_dir}")
    print(f"Episodes to generate: {args.num_episodes}")

    try:
        # Initialize generator (FrankaArm will handle ROS node initialization)
        generator = ThreePhaseDataGenerator(
            args.base_dataset_dir,
            args.new_dataset_dir,
            args.push_dataset_dir,
            args.push_start_frame,
            pos_x_range=args.pos_x_range,
            pos_y_range=args.pos_y_range,
            pos_z_range=args.pos_z_range,
            debug=args.debug
        )

        print(f"Loaded {len(generator.base_episodes)} pull drawer episodes")
        print(f"Loaded {len(generator.push_episodes)} push drawer episodes")
        print("Ready to start episode generation.")

        # Generate episodes with new flow: episode -> save -> user input -> robot init
        successful_episodes = 0

        # Initialize robot for first episode
        generator.robot_init_for_episode()

        for i in range(args.num_episodes):
            print(f"\n=== Episode {i+1}/{args.num_episodes} ===")

            # Ask user to start episode (robot is already at HOME_POSE)
            #input("[INFO] Press enter to start episode generation")

            # Select base episode
            if args.random_base_episodes:
                base_episode_idx = np.random.choice(list(generator.base_episodes.keys()))
            else:
                base_episode_idx = args.base_episode

            # Select push episode
            push_episode_idx = None
            if len(generator.push_episodes) > 0:
                if args.random_push_episodes:
                    push_episode_idx = np.random.choice(list(generator.push_episodes.keys()))
                elif args.push_episode is not None:
                    push_episode_idx = args.push_episode

            print(f"Using pull drawer episode {base_episode_idx} for generation")
            if push_episode_idx is not None:
                print(f"Using push drawer episode {push_episode_idx} starting from frame {args.push_start_frame} for reset")
            else:
                print("No push drawer episode selected - manual reset required")

            # Generate episode (Phase 0 + Phase 1 + Phase 2 + Phase 3 + Phase 4 + Save)
            success = generator.generate_episode(base_episode_idx, push_episode_idx)

            if success:
                successful_episodes += 1
                print(f"✅ Episode {i+1} completed successfully")

                # Ask user for next episode (after data is saved, before moving to home)
                if i < args.num_episodes - 1:  # Not the last episode
                    #input("[INFO] Press enter to continue to next episode")
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
        # Clean up resources
        try:
            if 'generator' in locals():
                generator.cleanup()
        except:
            pass
        print("Shutting down...")


if __name__ == "__main__":
    main()
