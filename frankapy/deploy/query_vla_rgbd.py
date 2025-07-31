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
    parser.add_argument('--instructions', type=str, default="test")
    parser.add_argument('--ctrl_freq', type=float, default=5.0)
    parser.add_argument('--record_dir', type=str, default='logs/openpi')
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--vla_server_ip', type=str, default='localhost', help='The IP address of the VLA server')
    parser.add_argument('--vla_server_port', type=int, default=9876, help='The port of the VLA server')
    return parser.parse_args()

class VLADeploy:
    """
    different from query_vla.py, this script is used to deploy VLA/affordance models with RGBD images
    obsevervation_window: add RGBD images
    action: absolute action in the form of [x, y, z, roll, pitch, yaw, gripper]
    """
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

    # maybe can be used for aligning with training
    def _jpeg_mapping(self, img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    def update_observation_window(self):
        images = self.camera.get_rgbd()
        # image = self.camera.get_rgb()[0] # get first camera rgb image, shape(height, width ,3)
        self.observation_window.append({
            'ee_pose_T': self.robot.get_pose().matrix, # np shape (4,4)
            'joints': self.robot.get_joints(), # np shape (7,)
            'gripper_width': np.array([self.robot.get_gripper_width()]), # np shape(1,)
            'instruction': self.args.instructions, # string
            'images': images.astype(np.uint8) # support multi camera
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
        # start a new skill 
        self.robot.goto_pose(FC.HOME_POSE, duration=10, dynamic=True, buffer_time=100000000, skill_desc='MOVE', 
                        cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES, ignore_virtual_walls = True)

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
                            "ee_pose_T": observation['ee_pose_T'],
                            "joints": observation['joints'],
                            "gripper_width": observation['gripper_width'],
                            "images": observation['images'], 
                            "instruction": observation['instruction'],
                            }
                    ).json()
                    action = np.array(action['actions'])
                    if len(action.shape) == 1:
                        self.actions_list.append(action)
                    else:
                        for idx in range(action.shape[0]):
                            self.actions_list.append(action[idx])
                
                action = self.actions_list.pop(0) # affordance models e.g. GFlow returns absolute action
                print("request and inference time cost", time.time() - t1, "| action.shape", action.shape)
                timestamp = rospy.Time.now().to_time()-self.init_time
                # Compute target pose (need to check)
                self.command_xyz = action[:3]
                self.command_rotation = action[3:6]

                try:
                    self.command_transform = RigidTransform(rotation=self.command_rotation, translation=self.command_xyz, from_frame='franka_tool', to_frame='world')
                    self.robot.goto_pose(self.command_transform)
                    time.sleep(0.5)
                    
                except Exception as e:
                    if e is KeyboardInterrupt:
                        self.robot.stop_skill()
                        print(f"[WARN] Keyboard Interp : {e}")
                        break
                    self.ee_pose_init()
                    control_rate.sleep()
                    print(f"[WARN] Move failed? : {e}")
                    continue

                print(f"[STEP {step}] xyz: {self.command_xyz}, delta_euler: {self.command_rotation}, gripper: {gripper}")
                step += 1
                control_rate.sleep()
        except Exception as e:
            self.robot.stop_skill()
            control_rate.sleep()
            print(f"[WARN] Keyboard Interp : {e}")

        self.robot.stop_skill()
        rospy.loginfo('Done')
        print("[INFO] Reaching Max-steps, Inference loop finished.")

def main():
    args = parse_arguments()
    timestamp = time.strftime("GFlow-%Y-%m-%d-%H-%M-%S")
    args.record_dir = os.path.join(args.record_dir, timestamp)
    os.makedirs(args.record_dir, exist_ok=True)

    # Save arguments and git commit
    with open(os.path.join(args.record_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    os.system(f'git rev-parse HEAD > {os.path.join(args.record_dir, "git_commit.txt")}')

    agent = VLADeploy(args)
    agent.robot_init()
    agent.run_inference_loop()

if __name__ == '__main__':
    main()


