import os
import sys
import time
import json
import torch
import cv2
import requests
import argparse
import json_numpy
import numpy as np
from pathlib import Path
from collections import deque
from PIL import Image as PImage
from transforms3d.euler import quat2euler, euler2quat
from autolab_core import RigidTransform
from frankapy import FrankaArm
json_numpy.patch()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instructions', type=str, required=True)
    parser.add_argument('--openvla_unnorm_key', type=str, default='default')
    parser.add_argument('--ip_address', type=str, default='localhost')
    parser.add_argument('--ctrl_freq', type=float, default=5.0)
    parser.add_argument('--record_dir', type=str, default='logs/openvla')
    parser.add_argument('--use_target_delta', action='store_true', help='Use target delta from observation or not')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--vla_server_ip', type=str, default='0.0.0.0', help='The IP address of the VLA server')
    parser.add_argument('--vla_server_port', type=int, default=9000, help='The port of the VLA server')
    parser.set_defaults(use_target_delta=True)
    return parser.parse_args()

class OpenVLADeploy:
    def __init__(self, args):
        self.args = args
        self.observation_window = deque(maxlen=2)

        # Interfaces
        self.robot = FrankaArm()
        self.camera = CameraInterface(ip_address=args.ip_address)

        # Record settings
        self.record_dir = args.record_dir
        os.makedirs(self.record_dir, exist_ok=True)

        self.max_steps = args.max_steps
        self.ctrl_freq = args.ctrl_freq
        self.time_step = 1.0 / self.ctrl_freq
        self.act_url = f"http://{args.vla_server_ip}:{args.vla_server_port}/act"


    # maybe can be used for aligning with training
    def _jpeg_mapping(self, img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    def update_observation_window(self):
        image, _ = self.camera.read_once()
        self.observation_window.append({
            'instruction': self.args.instructions,
            'image': image
        })

    def run_inference_loop(self):
        step = 0
        self.robot.go_home()
        input("[INFO] Press enter to continue")
        self.robot.go_home()
        time.sleep(1)

        pre_abs_xyz, pre_abs_quat = self.robot.get_ee_pose()
        pre_abs_euler = quat2euler(pre_abs_quat)

        print("[INFO] Starting inference loop...")
        while step < self.max_steps:
            t0 = time.time()

            self.update_observation_window()
            observation = self.observation_window[-1]

            # request and inference
            t1 = time.time()
            action = requests.post(
                self.act_url,
                json={
                    "image": observation['image'].as_type(np.uint8), 
                    "instruction": observation['instruction'],
                    }
            ).json()
            action = np.array(action)
            print("request and inference time cost", time.time() - t1, "|action.shape", action.shape)

            # robot control
            if not self.args.use_target_delta:
                pre_abs_xyz, pre_abs_quat = self.robot.get_ee_pose()
                pre_abs_euler = quat2euler(pre_abs_quat)

            delta_xyz, delta_euler, gripper = action[:3], action[3:6], action[-1]
            target_xyz = pre_abs_xyz + delta_xyz
            target_euler = pre_abs_euler + delta_euler

            try:
                self.robot.update_desired_ee_pose(
                    torch.from_numpy(target_xyz).float(),
                    torch.from_numpy(euler2quat(target_euler)).float(),
                )
                # self.robot.move_to_ee_pose(
                #     torch.from_numpy(target_xyz).float(),
                #     torch.from_numpy(euler2quat(target_euler)).float(),
                #     blocking=True,
                #     timeout=10.0
                # )
                self.gripper.goto(gripper, speed=0.1, force=0.1)
            except Exception as e:
                print(f"[WARN] Move failed: {e}")
                continue

            print(f"[STEP {step}] delta_xyz: {delta_xyz}, delta_euler: {delta_euler}, gripper: {gripper}")
            pre_abs_xyz, pre_abs_euler = target_xyz, target_euler
            step += 1

            duration = time.time() - t0
            if duration < self.time_step:
                time.sleep(self.time_step - duration)

        print("[INFO] Inference loop finished.")

def main():
    args = parse_arguments()
    timestamp = time.strftime("OpenVLA-%Y-%m-%d-%H-%M-%S")
    args.record_dir = os.path.join(args.record_dir, timestamp)
    os.makedirs(args.record_dir, exist_ok=True)

    # Save arguments and git commit
    with open(os.path.join(args.record_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    os.system(f'git rev-parse HEAD > {os.path.join(args.record_dir, "git_commit.txt")}')

    agent = OpenVLADeploy(args)
    agent.run_inference_loop()

if __name__ == '__main__':
    main()


