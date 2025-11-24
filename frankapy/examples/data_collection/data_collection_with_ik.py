"""
Real robot data collection script that uses IK to convert EE pose commands to joint commands.
This aligns the low-level control interface between real robot and simulation.
"""
import os
import sys
import time
import json
import tyro
import copy
import rospy
import numpy as np
from dataclasses import dataclass
from transforms3d.euler import euler2quat, euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat
import torch

# Real robot imports
from space_mouse_wrapper.space_mouse import SpaceMouse
from realsense_wrapper.realsense_d435 import RealsenseAPI
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy.franka_constants import FrankaConstants as FC
from examples.data_collection.vla_data_collector import VLADataCollector
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, CartesianImpedanceSensorMessage
from kinematics import PANDA_URDF_PATH
from kinematics.panda_ik_solver_sim_aligned import create_sim_aligned_ik_solver, SimPose # Use the simulation-aligned Panda IK Solver


class RealDataCollectionWithIK:
    """Data collection class that uses IK to convert EE pose to joint positions.
    
    Uses SimAlignedPandaIKSolver - fully aligned with ManiSkill simulation kinematics.
    """
    
    def __init__(self, args, robot: FrankaArm, cameras: RealsenseAPI, use_space_mouse: bool=False):
        self.robot: FrankaArm = robot
        self.cameras: RealsenseAPI = cameras

        self.args: Args = args
        self.data_collector = VLADataCollector(robot, cameras)
        self.use_space_mouse = use_space_mouse
        if use_space_mouse:
            self.space_mouse = SpaceMouse(vendor_id=0x256f, product_id=0xc635)

        self.episode_idx = 0
        self.action_steps = 0
        self.instruction = args.instruction
        self.init_xyz = None
        self.init_rotation = None
        self.command_xyz = None
        self.command_rotation = None
        self.control_frequency = 5
        self.control_time_step = 1.0/self.control_frequency
        self.last_gripper_width = None
        self.init_time = rospy.Time.now().to_time()

        self.pos_scale = args.pos_scale
        self.rot_scale = args.rot_scale

        # Initialize IK solver
        self._init_ik_solver()

    def _init_ik_solver(self):
        """Initialize the simulation-aligned Panda IK solver."""
        # Panda URDF path relative to ManiSkill
        urdf_path = PANDA_URDF_PATH
        
        # Use CPU for IK solver (more stable for real-time control)
        device = "cuda" if self.args.use_gpu_ik else "cpu"
        
        print(f"[INFO] Initializing Simulation-Aligned Panda IK solver on device: {device}")
        self.ik_solver = create_sim_aligned_ik_solver(
            urdf_path=urdf_path,
            device=device
        )
        print(f"[INFO] IK solver initialized successfully")
        print(f"[INFO] Solver mode: {self.ik_solver.alignment_mode}")
        print(f"[INFO] Max iterations: {self.ik_solver.max_iterations}")

    def _compute_ik_for_pose(self, position, rotation_matrix, initial_joints=None):
        """
        Compute IK for given pose using the simulation-aligned Panda IK solver.
        
        Args:
            position: np.array of shape (3,) - xyz position
            rotation_matrix: np.array of shape (3,3) - rotation matrix
            initial_joints: np.array of shape (7,) - initial joint positions (warm start)
            
        Returns:
            np.array of shape (7,) - joint positions, or None if IK fails
        """
        # Create target pose using SimPose (aligned with ManiSkill)
        # Convert rotation matrix to quaternion (wxyz format) using transforms3d
        quat_wxyz = mat2quat(rotation_matrix)  # transforms3d returns [w,x,y,z]
        
        target_pose = SimPose.from_pq(
            position=position,
            quaternion=quat_wxyz,
            device=self.ik_solver.device
        )
        
        # Solve IK with warm start
        ik_solution = self.ik_solver.compute_ik(
            target_pose,
            initial_qpos=initial_joints  # Warm start for better convergence
        )
        
        if ik_solution is not None:
            # Return numpy array (remove batch dimension)
            return ik_solution[0].cpu().numpy()
        else:
            return None

    def ee_pose_init(self):
        """Initialize the end-effector pose."""
        time.sleep(0.5)
        pose = self.robot.get_pose()
        self.init_xyz = pose.translation
        self.init_rotation = pose.rotation
        self.command_xyz = self.init_xyz.copy()
        self.command_rotation = self.init_rotation.copy()
        
        # Get initial joint positions for IK warm start
        self.current_joints = self.robot.get_joints()
        print(f"[INFO] Initial joints: {self.current_joints}")
        print(f"[INFO] Initial EE position: {self.init_xyz}")

    def _apply_control_data_clip_and_scale(self, control_tensor, offset=0.0):
        """Clip and scale control data."""
        control_tensor = np.clip(control_tensor, -1.0, 1.0)
        scaled_tensor = np.zeros_like(control_tensor)
        positive_mask = (control_tensor >= offset)
        negative_mask = (control_tensor <= -offset)
        if offset < 1.0 and offset >= 0.0:
            scaled_tensor[positive_mask] = (control_tensor[positive_mask] - offset) / (1.0 - offset)
            scaled_tensor[negative_mask] = (control_tensor[negative_mask] + offset) / (1.0 - offset)
        else:
            raise ValueError(f"offset should in the range of 0-1, while the offset is set to be {offset}.")
        return np.clip(scaled_tensor, -1.0, 1.0)

    def collect_data(self):
        """Main data collection loop using IK-based joint control."""
        input("Press enter to start collection with IK-based control")
        print("[INFO] Starting data collection with IK solver...")
        control_rate = rospy.Rate(self.control_frequency)
        
        # Statistics
        ik_success_count = 0
        ik_failure_count = 0
        
        try:
            while True:
                try:
                    # Read SpaceMouse controls
                    if self.use_space_mouse:
                        control = self.space_mouse.control
                        control_gripper = self.space_mouse.gripper_status
                        control_quit = self.space_mouse.quit_signal
                    else:
                        control = np.zeros((6,))
                        control_gripper = 1
                        control_quit = False

                    if control_quit:
                        print("[INFO] Data collection stopped by user.")
                        self.robot.stop_skill()
                        rospy.loginfo('Done')
                        break

                    # Process control signals
                    control_xyz = control[:3]
                    if self.args.user_frame:
                        control_xyz[:2] *= -1

                    control_euler = control[3:6][[1,0,2]] * np.array([-1,-1,1])
                    control_xyz = self._apply_control_data_clip_and_scale(control_xyz, 0.35)
                    control_euler = self._apply_control_data_clip_and_scale(control_euler, 0.35)

                    # Compute delta pose
                    delta_xyz = control_xyz * self.pos_scale
                    delta_euler = control_euler * self.rot_scale
                    delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz')

                    # Update command pose
                    self.command_xyz += delta_xyz
                    self.command_rotation = np.matmul(self.command_rotation, delta_rotation)

                    timestamp = rospy.Time.now().to_time() - self.init_time

                    # Compute IK to get target joint positions
                    target_joints = self._compute_ik_for_pose(
                        self.command_xyz,
                        self.command_rotation,
                        initial_joints=self.current_joints
                    )

                    if target_joints is None:
                        print(f"[WARNING] IK failed at step {self.action_steps}, skipping this command")
                        ik_failure_count += 1
                        control_rate.sleep()
                        continue
                    
                    ik_success_count += 1
                    
                    # Verify IK solution (optional, for debugging)
                    if self.args.verify_ik and self.action_steps % 10 == 0:
                        # Create target pose for verification using transforms3d
                        quat_wxyz = mat2quat(self.command_rotation)  # returns [w,x,y,z]
                        target_pose_for_verify = SimPose.from_pq(
                            position=self.command_xyz,
                            quaternion=quat_wxyz,
                            device=self.ik_solver.device
                        )
                        pos_error, ori_error = self.ik_solver.verify_ik_solution(
                            torch.tensor([target_joints], device=self.ik_solver.device),
                            target_pose_for_verify
                        )
                        print(f"[DEBUG] Step {self.action_steps}: "
                              f"pos_err={pos_error*1000:.3f}mm, "
                              f"ori_err={np.rad2deg(ori_error):.3f}deg")

                    # Save action data
                    save_action = {
                        "delta": {
                            "position": delta_xyz,
                            "orientation": euler2quat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz'),
                            "euler_angle": delta_euler,
                        },
                        "abs": {
                            "position": copy.deepcopy(self.command_xyz),
                            "euler_angle": np.array([mat2euler(self.command_rotation, 'sxyz')])[0],
                            "joints": target_joints.copy(),  # Save computed joint positions
                        },
                        "gripper_width": control_gripper
                    }

                    # Collect data
                    self.data_collector.update_data_dict(
                        instruction=self.instruction,
                        action=save_action,
                        timestamp=timestamp,
                    )

                    gripper_width = FC.GRIPPER_WIDTH_MAX * control_gripper

                    # Send joint command to robot using dynamic joint control
                    joint_proto_msg = JointPositionSensorMessage(
                        id=self.action_steps + 1,
                        timestamp=timestamp,
                        joints=target_joints.tolist()
                    )
                    
                    # Use joint impedance controller
                    ros_pub_sensor_msg = make_sensor_group_msg(
                        trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                            joint_proto_msg, SensorDataMessageType.JOINT_POSITION
                        )
                    )
                    
                    rospy.loginfo(f'Publishing: Step {self.action_steps+1}, IK joints sent')
                    self.robot.publish_sensor_values(ros_pub_sensor_msg)
                    
                    # Update current joints for next IK warm start
                    self.current_joints = target_joints

                    # Control gripper
                    if abs(gripper_width - self.last_gripper_width) > 0.02:
                        grasp = True if control_gripper < 0.5 else False
                        self.robot.goto_gripper(
                            gripper_width, 
                            grasp=grasp, 
                            force=30, 
                            speed=0.12,
                            block=False, 
                            skill_desc="control_gripper"
                        )
                        self.last_gripper_width = gripper_width

                    self.action_steps += 1
                    control_rate.sleep()
                    
                except KeyboardInterrupt:
                    print("[INFO] Data collection stopped by keyboard interrupt.")
                    self.robot.stop_skill()
                    rospy.loginfo('Done')
                    break
                except Exception as e:
                    print(f"[ERROR] An error occurred during data collection: {e}")
                    import traceback
                    traceback.print_exc()
                    control_rate.sleep()
                    self.ee_pose_init()
                    continue
        finally:
            # Print statistics
            total_ik = ik_success_count + ik_failure_count
            if total_ik > 0:
                success_rate = 100.0 * ik_success_count / total_ik
                print(f"\n[INFO] IK Statistics:")
                print(f"  Success: {ik_success_count}")
                print(f"  Failure: {ik_failure_count}")
                print(f"  Success Rate: {success_rate:.2f}%")
            
            # Clean up resources
            if self.use_space_mouse:
                self.space_mouse.close()
            if hasattr(self, 'cameras'):
                self.cameras.close()

    def get_next_episode_idx(self, task_dir):
        """Find the next episode index by identifying the highest existing episode number."""
        if not os.path.exists(task_dir):
            return 0

        all_items = os.listdir(task_dir)
        episode_dirs = [item for item in all_items 
                       if os.path.isdir(os.path.join(task_dir, item)) and item.startswith("episode_")]

        if not episode_dirs:
            return 0

        episode_numbers = []
        for dir_name in episode_dirs:
            try:
                episode_number = int(dir_name.split("_")[1])
                episode_numbers.append(episode_number)
            except (IndexError, ValueError):
                continue

        if not episode_numbers:
            return 0

        return max(episode_numbers) + 1

    def save_data(self, task_dir):
        """Save collected data to disk."""
        if self.action_steps > self.args.max_action_steps:
            print("action_steps too large, data not saved")
            return False
            
        if self.args.episode_idx < 0:
            self.episode_idx = self.get_next_episode_idx(task_dir)
        else:
            self.episode_idx = self.args.episode_idx
            
        episode_dir = os.path.join(task_dir, f"episode_{self.episode_idx}")

        os.makedirs(episode_dir, exist_ok=True)
        metadata_path = os.path.join(episode_dir, "metadata.json")
        self.data_collector.save_data(episode_dir, self.episode_idx)

        metadata = {
            "task_name": self.args.task_name,
            "episode_idx": self.episode_idx,
            "action_steps": self.action_steps,
            "instruction": self.instruction,
            "control_type": "ik_based_joint_control",
            "ik_solver": "SimAlignedPandaIKSolver",
            "ik_solver_mode": self.ik_solver.alignment_mode,
            "ik_max_iterations": self.ik_solver.max_iterations,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[INFO] Data saved to {episode_dir}")
        print(f"[INFO] Metadata saved to {metadata_path}")
        return True


@dataclass 
class Args:
    """Data collection script arguments."""
    task_name: str  # Task name for the dataset
    instruction: str  # Instruction for data collection
    dataset_dir: str = "datasets"  # Directory to save dataset
    min_action_steps: int = 200  # Minimum action_steps for data collection
    max_action_steps: int = 1000  # Maximum action_steps for data collection  
    episode_idx: int = -1  # Episode index to save data (-1 for auto-increment)
    user_frame: bool = False  # Use user frame
    pos_scale: float = 0.015  # The scale of xyz action
    rot_scale: float = 0.025  # The scale of rotation action
    use_gpu_ik: bool = False  # Use GPU for IK solver
    verify_ik: bool = False  # Verify IK solution with FK (for debugging)


def main():
    args = tyro.cli(Args)
    robot = FrankaArm()
    cameras = RealsenseAPI()
    collection = RealDataCollectionWithIK(args, robot, cameras, use_space_mouse=True)
    
    # Home
    robot.reset_joints()
    robot.open_gripper()
    
    # Start a new skill with joint control
    # First get home joints
    home_joints = robot.get_joints()
    print(f"[INFO] Home joints: {home_joints}")
    
    # Start dynamic joint control mode
    robot.goto_joints(
        home_joints, 
        duration=10, 
        dynamic=True, 
        buffer_time=100000000, 
        skill_desc='JOINT_CONTROL'
    )
    
    collection.last_gripper_width = robot.get_gripper_width()
    collection.ee_pose_init()
    collection.collect_data()

    if collection.action_steps < args.min_action_steps:
        print(f"[Error] Save failure (#step < {args.min_action_steps}), please check your cameras and arms and try again.")
        exit(-1)

    task_dir = os.path.join(args.dataset_dir, args.task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    # Save data
    result = collection.save_data(task_dir)
    if result:
        print(f"\033[32m\nSave success, saved {collection.action_steps} action_steps of data.\033[0m\n")


if __name__ == "__main__":
    main()
