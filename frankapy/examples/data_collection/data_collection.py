import os
import sys
import time
import json
import torch
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from space_mouse_wrapper.space_mouse import SpaceMouse
from realsense_wrapper.realsense_d435 import RealsenseAPI
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy.franka_constants import FrankaConstants as FC
from examples.data_collection.vla_data_collector import VLADataCollector
from transforms3d.euler import euler2quat, euler2mat, mat2euler
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage
import rospy
import copy

class RealDataCollection:
    def __init__(self, args, robot: FrankaArm, cameras: RealsenseAPI, use_space_mouse: bool=False):
        self.robot: FrankaArm = robot
        self.cameras: RealsenseAPI = cameras

        self.args = args
        self.data_collector = VLADataCollector(robot, cameras)
        self.use_space_mouse = use_space_mouse
        if use_space_mouse:
            self.space_mouse = SpaceMouse(vendor_id=0x256f, product_id=0xc635)

        self.episode_idx = 0  # Default episode index
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

    def ee_pose_init(self):
        time.sleep(0.5)
        pose = self.robot.get_pose()
        self.init_xyz = pose.translation
        self.init_rotation = pose.rotation
        self.command_xyz = self.init_xyz
        self.command_rotation = self.init_rotation

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
        print("[INFO] Starting data collection...")
        control_rate = rospy.Rate(self.control_frequency)
        while True:
            try:
                # Read SpaceMouse controls
                if self.use_space_mouse:
                    control = self.space_mouse.control # xyz in range [-1, 1] in m,  roll picth yaw [-1, 1] in deg
                    control_gripper = self.space_mouse.gripper_status
                    control_quit = self.space_mouse.quit_signal
                else:
                    control = np.zeros((6,))
                    control_gripper = 1 # [0,1]
                    control_quit = False

                if control_quit:
                    print("[INFO] Data collection stopped by user.")
                    self.robot.stop_skill()
                    rospy.loginfo('Done')
                    break

                # for space mouse: roll pitch yaw -> For panda: pitch roll yaw (defined by user bingwen)
                control_xyz = control[:3]
                control_euler = control[3:6][[1,0,2]] * np.array([-1,-1,1])
                control_xyz = self._apply_control_data_clip_and_scale(control_xyz, 0.35)
                control_euler = self._apply_control_data_clip_and_scale(control_euler, 0.35)

                delta_xyz = control_xyz * 0.015
                # delta_xyz *= 0
                delta_euler = control_euler * 0.025 # z, y, x
                # delta_euler *= 0
                delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2],'sxyz')

                # Compute target pose
                self.command_xyz += delta_xyz
                self.command_rotation = np.matmul(self.command_rotation, delta_rotation)
                # u, _, vh = np.linalg.svd(self.command_rotation)
                # self.command_rotation = np.matmul(u, vh)

                timestamp = rospy.Time.now().to_time()-self.init_time

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
                    "gripper_width": control_gripper
                }

                # Collect data
                self.data_collector.update_data_dict(
                    instruction=self.instruction,
                    action=save_action,
                    # xyz=delta_xyz,
                    # quat=euler2quat(delta_euler[0],delta_euler[1],delta_euler[2],'sxyz'),
                    # gripper_width=control_gripper,
                    timestamp=timestamp,
                )

                self.command_transform = RigidTransform(rotation=self.command_rotation, translation=self.command_xyz, from_frame='franka_tool', to_frame='world')
                gripper_width = FC.GRIPPER_WIDTH_MAX * control_gripper
                # control joint and gripper

                # ros data send and control
                pub_traj_gen_proto_msg = PosePositionSensorMessage(
                    id=self.action_steps+1, timestamp=timestamp, 
                    position=self.command_transform.translation, quaternion=self.command_transform.quaternion
                )
                fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                    id=self.action_steps+1, timestamp=timestamp,
                    translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES,
                    rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
                )
                ros_pub_sensor_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        pub_traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                    feedback_controller_sensor_msg=sensor_proto2ros_msg(
                        fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
                )
                rospy.loginfo(f'Publishing: Steps {self.action_steps+1}, delta_xyz = {delta_xyz}')
                self.robot.publish_sensor_values(ros_pub_sensor_msg)
                if abs(gripper_width - self.last_gripper_width) > 0.02:
                    grasp = True if control_gripper<0.5 else False
                    self.robot.goto_gripper(gripper_width, grasp=grasp, force=FC.GRIPPER_MAX_FORCE/3.0, speed=0.12, block=False, skill_desc="control_gripper")
                    self.last_gripper_width = gripper_width

                self.action_steps += 1
                control_rate.sleep()
            except KeyboardInterrupt:
                print("[INFO] Data collection stopped by user.")
                self.robot.stop_skill()
                rospy.loginfo('Done')
                break
            except Exception as e:
                print(f"[ERROR] An error occurred during data collection: {e}")
                control_rate.sleep()
                # time.sleep(self.control_time_step)
                self.ee_pose_init()
                continue
                # break

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
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[INFO] Data saved to {episode_dir}")
        print(f"[INFO] Metadata saved to {metadata_path}")
        return True


def get_arguments():
    parser = argparse.ArgumentParser(description="Data collection script.")
    parser.add_argument("--dataset_dir", type=str, default="datasets", help="Directory to save dataset.")  # Default to "datasets"
    parser.add_argument("--task_name", type=str, required=True, help="Task name for the dataset.")
    parser.add_argument("--min_action_steps", type=int, default=200, help="Minimum action_steps for data collection.")
    parser.add_argument("--max_action_steps", type=int, default=1000, help="Maximum action_steps for data collection.")
    parser.add_argument("--episode_idx", type=int, default=-1, help="Episode index to save data (-1 for auto-increment).")
    parser.add_argument("--instruction", type=str, required=True, help="Instruction for data collection.")
    return parser.parse_args()



def main():
    args = get_arguments()
    robot = FrankaArm()
    cameras = RealsenseAPI()
    collection = RealDataCollection(args, robot, cameras, use_space_mouse=True)
    # Home
    robot.reset_joints()
    robot.open_gripper()
    # start a new skill 
    robot.goto_pose(FC.HOME_POSE, duration=10, dynamic=True, buffer_time=100000000, skill_desc='MOVE', 
                    cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES, ignore_virtual_walls = True)
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
        print(f"\033[32m\nSave success, save {collection.action_steps} action_steps of data.\033[0m\n")


if __name__ == "__main__":
    main()
