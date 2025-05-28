import os
import sys
import time
import json
import torch
import cv2
import rospy
import requests
import argparse
import json_numpy
import numpy as np
from pathlib import Path
from collections import deque
from PIL import Image as PImage
from transforms3d.euler import euler2quat, euler2mat, quat2euler
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from realsense_wrapper.realsense_d435 import RealsenseAPI
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.franka_constants import FrankaConstants as FC
from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage
json_numpy.patch()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--ctrl_freq', type=float, default=5.0)
    return parser.parse_args()

class DataRepleyer:
    """
    Load dataset from a directory
    Replay it
    """

    def __init__(self, args):
        self.args = args
        self.dataset_dir = self.args.dataset_dir

        self.data = self.load_data_from_dir(self.dataset_dir)

        self.ctrl_freq = args.ctrl_freq

        self.last_close = -1

        # Interfaces
        self.robot = FrankaArm()
        self.camera = RealsenseAPI()

        self.init_time = rospy.Time.now().to_time()
    
    def data_structure_check(self, data):
        for key, item in data.items():
            if isinstance(item, dict):
                print(key, ":")
                self.data_structure_check(item)
            elif isinstance(item, np.ndarray):
                print(key, ":", item.shape)
            else:
                print(key, ":", item)

    def load_data_from_dir(self, dataset_dir):
        data = np.load(os.path.join(dataset_dir, "data.npy"), allow_pickle=True).item()
        self.data_structure_check(data)

        return data

    def ee_pose_init(self):
        time.sleep(0.5)
        pose = self.robot.get_pose()
        self.init_xyz = pose.translation
        self.init_rotation = pose.rotation
        self.command_xyz = self.init_xyz
        self.command_rotation = self.init_rotation

    def robot_init(self):
        self.robot.reset_joints()
        self.robot.open_gripper()
        input("[INFO] Press enter to continue")
        # start a new skill 
        self.robot.goto_pose(FC.HOME_POSE, duration=10, dynamic=True, buffer_time=100000000, skill_desc='MOVE', 
                        cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES, ignore_virtual_walls = True)

    def replay_action(self):
        position_sequence = self.data["action"]["end_effector"]["delta_position"]
        rotation_sequence = self.data["action"]["end_effector"]["delta_euler"]
        gripper_width_sequence = self.data["action"]["end_effector"]["gripper_width"]

        step = 0
        self.ee_pose_init()
        control_rate = rospy.Rate(self.ctrl_freq)
        print(f"[INFO] Starting replay data with {position_sequence.shape[0]} steps...")

        replay_step = position_sequence.shape[0]

        for i in range(replay_step):
            pos = position_sequence[i]
            rot = rotation_sequence[i]
            gripper_width = gripper_width_sequence[i]
            action = np.array([
                pos[0], pos[1], pos[2],
                rot[0], rot[1], rot[2],
                gripper_width
            ])

            timestamp = rospy.Time.now().to_time()-self.init_time
            delta_xyz, delta_euler, gripper = action[:3], action[3:6], action[-1] # need to check

            # if gripper < 0.5:
            #     if self.last_close == -1:
            #         self.last_close = 3
            #     elif self.last_close > 0:
            #         self.last_close -= 1
                
            #     if self.last_close == 0:
            #         gripper = 1

            delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2],'sxyz')
            # Compute target pose
            self.command_xyz += delta_xyz
            self.command_rotation = np.matmul(self.command_rotation, delta_rotation)
            try:
                self.command_transform = RigidTransform(rotation=self.command_rotation, translation=self.command_xyz, from_frame='franka_tool', to_frame='world')
                gripper_width = FC.GRIPPER_WIDTH_MAX * gripper

                # ros data send and control
                pub_traj_gen_proto_msg = PosePositionSensorMessage(
                    id=step+1, timestamp=timestamp, 
                    position=self.command_transform.translation, quaternion=self.command_transform.quaternion
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
                # rospy.loginfo(f'Publishing: Steps {step+1}, delta_xyz = {delta_xyz}')
                self.robot.publish_sensor_values(ros_pub_sensor_msg)

                current_gripper_width = self.robot.get_gripper_width()
                if abs(gripper_width - current_gripper_width) > 0.01:
                    grasp = True if gripper<0.5 else False
                    self.robot.goto_gripper(gripper_width, grasp=grasp, force=FC.GRIPPER_MAX_FORCE/3.0, speed=0.12, block=False, skill_desc="control_gripper")

            except Exception as e:
                self.ee_pose_init()
                control_rate.sleep()
                print(f"[WARN] Move failed? : {e}")
                continue

            print(f"[STEP {step}] delta_xyz: {delta_xyz}, delta_euler: {delta_euler}, gripper: {gripper}")
            step += 1
            control_rate.sleep()

        self.robot.stop_skill()
        rospy.loginfo('Done')
        print("[INFO] Replay end.")

def main():
    args = parse_arguments()
    data_replayer = DataRepleyer(args)
    data_replayer.robot_init()
    data_replayer.replay_action()

if __name__=="__main__":
    main()