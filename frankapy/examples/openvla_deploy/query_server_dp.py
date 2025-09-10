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
    parser.add_argument('--ctrl_freq', type=float, default=5.0)
    parser.add_argument('--record_dir', type=str, default='logs/diffusion_policy')
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--vla_server_ip', type=str, default='localhost', help='The IP address of the DP server')
    parser.add_argument('--vla_server_port', type=int, default=9876, help='The port of the DP server')
    return parser.parse_args()

class DiffusionPolicyDeploy:
    def __init__(self, args):
        self.args = args
        self.observation_window = deque(maxlen=2)

        # Interfaces
        self.robot = FrankaArm()
        self.camera = RealsenseAPI()

        # Record settings
        self.record_dir = args.record_dir
        os.makedirs(self.record_dir, exist_ok=True)

        self.init_xyz = None
        self.init_rotation = None
        self.command_xyz = None
        self.command_rotation = None
        self.actions_list = []

        self.max_steps = args.max_steps
        self.ctrl_freq = args.ctrl_freq
        self.act_url = f"http://{args.vla_server_ip}:{args.vla_server_port}/act"
        self.init_time = rospy.Time.now().to_time()

    def get_robot_state(self):
        """Get current robot proprioceptive state"""
        # Get current pose
        pose = self.robot.get_pose()
        position = pose.translation  # (3,)
        quaternion = pose.quaternion  # (4,) [x, y, z, w]
        
        # Get gripper state
        gripper_width = self.robot.get_gripper_width()
        
        # Combine into 8D state: [pos(3), quat(4), gripper(1)]
        robot_state = np.concatenate([
            position,
            quaternion,
            [gripper_width]
        ]).astype(np.float32)
        
        return robot_state

    def update_observation_window(self):
        images = self.camera.get_rgb()  # Multi-camera images
        robot_state = self.get_robot_state()  # Get proprioceptive state
        
        self.observation_window.append({
            'images': images,
            'robot_state': robot_state
        })

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
        time.sleep(3.0)
        # start a new skill
        self.robot.goto_pose(FC.HOME_POSE, duration=10, dynamic=True, buffer_time=100000000, skill_desc='MOVE',
                             cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES, ignore_virtual_walls=True)

    def run_inference_loop(self):
        step = 0
        self.ee_pose_init()
        control_rate = rospy.Rate(self.ctrl_freq)
        print("[INFO] Starting inference loop...")
        try:
            while step < self.max_steps:
                self.update_observation_window()
                observation = self.observation_window[-1]

                if len(self.actions_list) == 0:
                    # request and inference
                    t1 = time.time()
                    action = requests.post(
                        self.act_url,
                        json={
                            "images": observation['images'].astype(np.uint8), 
                            "proprio_state": observation['robot_state'],  # Send robot state instead of instruction
                        }
                    ).json()
                    action = np.array(action)
                    if len(action.shape) == 1:
                        self.actions_list.append(action)
                    else:
                        for idx in range(action.shape[0]):
                            self.actions_list.append(action[idx])
                
                action = self.actions_list.pop(0)
                print("request and inference time cost", time.time() - t1, "| action.shape", action.shape)

                timestamp = rospy.Time.now().to_time()-self.init_time
                delta_xyz, delta_euler, gripper = action[:3], action[3:6], action[-1]
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
        except Exception as e:
            self.ee_pose_init()
            control_rate.sleep()
            print(f"[WARN] Keyboard Interp : {e}")

        self.robot.stop_skill()
        rospy.loginfo('Done')
        print("[INFO] Reaching Max-steps, Inference loop finished.")

def main():
    args = parse_arguments()
    timestamp = time.strftime("DiffusionPolicy-%Y-%m-%d-%H-%M-%S")
    args.record_dir = os.path.join(args.record_dir, timestamp)
    os.makedirs(args.record_dir, exist_ok=True)

    # Save arguments and git commit
    with open(os.path.join(args.record_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    os.system(f'git rev-parse HEAD > {os.path.join(args.record_dir, "git_commit.txt")}')

    agent = DiffusionPolicyDeploy(args)
    agent.robot_init()
    agent.run_inference_loop()

if __name__ == '__main__':
    main()
