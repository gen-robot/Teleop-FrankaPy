import numpy as np
import time
import copy
import json
import os
import tyro
import rospy
from dataclasses import dataclass

from space_mouse_wrapper.space_mouse import SpaceMouse
from realsense_wrapper.realsense_d435 import RealsenseAPI
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy.franka_constants import FrankaConstants as FC
from examples.data_collection.vla_data_collector import VLADataCollector
from transforms3d.euler import euler2quat, euler2mat, mat2euler
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage

# Import IK related classes
from kinematics.frankapy_utils import IKSolver, DynamicJointTrajectoryPublisher


class IKDataCollection:
    def __init__(self, args, robot: FrankaArm, cameras: RealsenseAPI, use_space_mouse: bool=False):
        self.robot: FrankaArm = robot
        self.cameras: RealsenseAPI = cameras

        self.args = args
        self.data_collector = VLADataCollector(robot, cameras)
        self.use_space_mouse = use_space_mouse
        if use_space_mouse:
            self.space_mouse = SpaceMouse(vendor_id=0x256f, product_id=0xc635)

        # Initialize IK solver and trajectory publisher
        self.ik_solver = IKSolver(
            urdf_path="./assets/panda/panda_v3.urdf",
            target_link_name="panda_hand_tcp"
        )
        self.traj_publisher = DynamicJointTrajectoryPublisher()

        self.episode_idx = 0  # Default episode index
        self.action_steps = 0
        self.instruction = args.instruction
        self.init_xyz = None
        self.init_rotation = None
        self.command_xyz = None
        self.command_rotation = None
        self.current_joints = None
        self.control_frequency = 5
        self.control_time_step = 1.0/self.control_frequency
        self.last_gripper_width = None
        self.init_time = rospy.Time.now().to_time()

        self.pos_scale = args.pos_scale
        self.rot_scale = args.rot_scale

        # Start dynamic execution with initial joint position
        self.dynamic_execution_started = False

    def ee_pose_init(self):
        time.sleep(0.5)
        pose = self.robot.get_pose()
        self.init_xyz = pose.translation
        self.init_rotation = pose.rotation
        self.command_xyz = self.init_xyz
        self.command_rotation = self.init_rotation
        self.last_gripper_width = self.robot.get_gripper_width()
        
        # Get current joint positions
        self.current_joints = self.robot.get_joints()
        print(f"[INFO] Initial pose: {pose}")
        print(f"[INFO] Initial joints: {self.current_joints}")

    def _apply_control_data_clip_and_scale(self, control_tensor, offset=0.0):
        control_tensor = np.clip(control_tensor, -1.0, 1.0)
        scaled_tensor = np.zeros_like(control_tensor)
        positive_mask = (control_tensor >= offset)
        negative_mask = (control_tensor <= -offset)
        if offset < 1.0 and offset>=0.0:
            scaled_tensor[positive_mask] = (control_tensor[positive_mask] - offset) / (1.0 - offset)
            scaled_tensor[negative_mask] = (control_tensor[negative_mask] + offset) / (1.0 - offset)
        else:
            raise ValueError(f"offset should in the range of 0-1, while the offset is set to be {offset}.")

        return np.clip(scaled_tensor, -1.0, 1.0)

    def collect_data(self):
        input("press enter to start collection")
        print("[INFO] Starting IK-based data collection...")
        control_rate = rospy.Rate(self.control_frequency)
        
        try:
            while True:
                try:    
                    # Read SpaceMouse controls
                    if self.use_space_mouse:
                        control = self.space_mouse.control # xyz in range [-1, 1] in m, roll picth yaw [-1, 1] in deg
                        control_gripper = self.space_mouse.gripper_status
                        control_quit = self.space_mouse.quit_signal
                    else:
                        # Keyboard or other control input would go here
                        control = np.zeros(6)
                        control_gripper = 1.0
                        control_quit = False

                    if control_quit:
                        print("[INFO] Quit signal received")
                        break

                    # Process control input similar to data_collection.py
                    control_xyz = control[:3]
                    if self.args.user_frame:
                        # Transform to user frame if needed
                        pass

                    # For space mouse: roll pitch yaw -> For panda: pitch roll yaw
                    control_euler = control[3:6][[1,0,2]] * np.array([-1,-1,1])
                    control_xyz = self._apply_control_data_clip_and_scale(control_xyz, 0.25)
                    control_euler = self._apply_control_data_clip_and_scale(control_euler, 0.25)

                    delta_xyz = control_xyz * self.pos_scale
                    delta_euler = control_euler * self.rot_scale
                    delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2],'sxyz')

                    # Compute target pose
                    self.command_xyz += delta_xyz
                    self.command_rotation = np.matmul(self.command_rotation, delta_rotation)

                    # Create target pose transform
                    target_pose = RigidTransform(
                        rotation=self.command_rotation, 
                        translation=self.command_xyz, 
                        from_frame='franka_tool', 
                        to_frame='world'
                    )

                    # Solve IK to get target joint positions
                    try:
                        target_joints = self.ik_solver.solve_ik(self.current_joints, target_pose)
                        print(f"[DEBUG] Target joints: {target_joints[:7]}")
                        print(f"[DEBUG] Joint delta: {target_joints[:7] - self.current_joints}")
                        
                        # Start dynamic execution on first iteration
                        # Start a new skill 
                        if not self.dynamic_execution_started:
                            total_duration = 1000  # Large duration for continuous execution
                            self.traj_publisher.start_dynamic_execution(
                                self.robot, target_joints, total_duration, buffer_time=10
                            )
                            self.dynamic_execution_started = True
                            time.sleep(0.1)  # Small delay after starting
                        
                        # Publish joint target
                        self.traj_publisher.publish_joint_target(target_joints, self.action_steps)
                        
                        # Update current joints for next iteration
                        self.current_joints = target_joints[:7]
                        
                    except Exception as e:
                        print(f"[ERROR] IK solving failed: {str(e)}")
                        continue

                    timestamp = rospy.Time.now().to_time() - self.init_time

                    # Save action data (same format as data_collection.py)
                    save_action = {
                        "delta": {
                            "position": delta_xyz,
                            "orientation": euler2quat(delta_euler[0],delta_euler[1],delta_euler[2],'sxyz'),
                            "euler_angle": delta_euler,
                        },
                        "abs": {
                            "position": copy.deepcopy(self.command_xyz),
                            "euler_angle": np.array([mat2euler(self.command_rotation, 'sxyz')])[0]
                        },
                        "gripper_width": control_gripper,
                        "target_joints": target_joints[:7]  # Add joint information
                    }

                    # Collect data
                    self.data_collector.update_data_dict(
                        instruction=self.instruction,
                        action=save_action,
                        timestamp=timestamp,
                    )

                    # Handle gripper control
                    gripper_width = FC.GRIPPER_WIDTH_MAX * control_gripper
                    if abs(gripper_width - self.last_gripper_width) > 0.01:  # Only move if significant change
                        try:
                            if gripper_width > self.last_gripper_width:
                                self.robot.open_gripper()
                            else:
                                self.robot.close_gripper()
                            self.last_gripper_width = gripper_width
                        except Exception as e:
                            print(f"[WARNING] Gripper control failed: {str(e)}")

                    self.action_steps += 1
                    
                    if self.action_steps % 50 == 0:
                        print(f"[INFO] Collected {self.action_steps} action steps")

                    control_rate.sleep()

                except KeyboardInterrupt:
                    print("[INFO] KeyboardInterrupt received, stopping data collection")
                    break
                except Exception as e:
                    print(f"[ERROR] Exception in data collection loop: {str(e)}")
                    continue
                    
        finally:
            # Terminate dynamic execution
            if self.dynamic_execution_started:
                self.traj_publisher.terminate_execution()
                print("[INFO] Dynamic execution terminated")
            
            # Clean up resources
            if self.use_space_mouse:
                self.space_mouse.close()
            if hasattr(self, 'cameras'):
                self.cameras.close()

    def get_next_episode_idx(self, task_dir):
        """
        Find the next episode index by identifying the highest existing episode number.

        Args:
            task_dir (str): The directory containing episode folders

        Returns:
            int: The next episode index (highest existing index + 1)
        """
        if not os.path.exists(task_dir):
            return 0  # Start with episode 0 if task directory doesn't exist

        # Get all items in the task directory
        all_items = os.listdir(task_dir)

        # Find all episode directories
        episode_dirs = []
        for item in all_items:
            item_path = os.path.join(task_dir, item)
            if os.path.isdir(item_path) and item.startswith("episode_"):
                episode_dirs.append(item)

        if not episode_dirs:
            return 0  # Start with episode 0 if no episode directories exist

        # Extract the episode numbers
        episode_numbers = []
        for dir_name in episode_dirs:
            try:
                # Extract number after "episode_"
                episode_number = int(dir_name.split("_")[1])
                episode_numbers.append(episode_number)
            except (IndexError, ValueError):
                # Skip directories that don't match the expected format
                continue

        if not episode_numbers:
            return 0  # If no valid episode numbers found, start with 0

        # Return the next episode index (max + 1)
        return max(episode_numbers) + 1

    def save_data(self, task_dir):
        if self.action_steps > self.args.max_action_steps:
            print("action_steps too large, data not saved")
            return False
        if self.args.episode_idx < 0:
            self.episode_idx = self.get_next_episode_idx(task_dir)
        else:
            self.episode_idx = self.args.episode_idx
        episode_dir = os.path.join(task_dir, f"episode_{self.episode_idx}")

        # Ensure save_dir exists
        os.makedirs(episode_dir, exist_ok=True)
        metadata_path = os.path.join(episode_dir, "metadata.json")
        self.data_collector.save_data(episode_dir, self.episode_idx)

        # Save metadata
        metadata = {
            "task_name": self.args.task_name,
            "episode_idx": self.episode_idx,
            "action_steps": self.action_steps,
            "instruction": self.instruction,
            "control_method": "ik_joints"  # Mark this as IK-based control
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[INFO] Data saved to {episode_dir}")
        print(f"[INFO] Metadata saved to {metadata_path}")
        return True


@dataclass
class Args:
    """IK-based data collection script arguments."""
    dataset_dir: str = "datasets"  # Directory to save dataset
    task_name: str  # Task name for the dataset
    min_action_steps: int = 200  # Minimum action_steps for data collection
    max_action_steps: int = 1000  # Maximum action_steps for data collection
    episode_idx: int = -1  # Episode index to save data (-1 for auto-increment)
    instruction: str  # Instruction for data collection
    user_frame: bool = False  # Use user frame
    pos_scale: float = 0.015  # The scale of xyz action
    rot_scale: float = 0.025  # The scale of rotation action


def main():
    args = tyro.cli(Args)
    robot = FrankaArm()
    cameras = RealsenseAPI()
    collection = IKDataCollection(args, robot, cameras, use_space_mouse=True)
    
    # Home
    robot.reset_joints()
    robot.open_gripper()
    
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
        print(f"\033[32m\nSave success, save {collection.action_steps} action_steps of data.\033[0m\n")


if __name__ == "__main__":
    main()