import os
import json
import time
import argparse
from collections import deque

import numpy as np
import requests
from transforms3d.euler import euler2quat

from franky import (
    Robot,
    Gripper,
    CartesianMotion,
    Affine,
    ReferenceType,
    RobotWebSession,
    TakeControlTimeoutError,
)
from realsense_wrapper.realsense_d435 import RealsenseAPI


def parse_common_args(default_record_dir: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructions", type=str, default="test")
    parser.add_argument("--ctrl_freq", type=float, default=5.0)
    parser.add_argument("--record_dir", type=str, default=default_record_dir)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--vla_server_ip", type=str, default="localhost")
    parser.add_argument("--vla_server_port", type=int, default=9876)
    parser.add_argument("--robot_host", type=str, default="172.16.0.12", help="FCI IP of the Franka robot")
    parser.add_argument(
        "--relative_dynamics",
        type=float,
        default=0.1,
        help="franky relative dynamics factor (0-1, smaller = safer and softer)",
    )
    parser.add_argument("--web_username", type=str, default="franka", help="Franka web UI username")
    parser.add_argument("--web_password", type=str, default="franka", help="Franka web UI password")
    return parser


def _quat_wxyz_to_xyzw(quat_wxyz):
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


class FrankyVLARunner:
    """Shared logic for OpenVLA/VLA control using franky (no ROS)."""

    def __init__(self, args, include_state: bool):
        self.args = args
        self.include_state = include_state
        self.robot = Robot(args.robot_host)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = args.relative_dynamics
        self.gripper = Gripper(args.robot_host)
        self.camera = RealsenseAPI()
        self.ctrl_dt = 1.0 / args.ctrl_freq
        self.actions_queue = deque()
        self.act_url = f"http://{args.vla_server_ip}:{args.vla_server_port}/act"
        self.start_time = time.time()
        self.web_session = None
        self._prepare_web_session()

    # -- Sensing -----------------------------------------------------------------
    def build_observation(self):
        images = self.camera.get_rgb().astype(np.uint8)
        obs = {"instruction": self.args.instructions, "images": images}
        if self.include_state:
            state = self.robot.state
            obs.update(
                {
                    # Affine.matrix is a 4x4 Eigen matrix; convert to list for JSON
                    "ee_pose_T": np.array(state.O_T_EE.matrix()).tolist(),
                    "joints": np.array(state.q).tolist(),
                    "gripper_width": [self.gripper.width],
                }
            )
        return obs

    # -- Network -----------------------------------------------------------------
    def request_actions(self, obs):
        resp = requests.post(self.act_url, json=obs)
        resp.raise_for_status()
        payload = resp.json()
        if self.include_state:
            payload = payload["actions"]
        actions = np.array(payload)
        if actions.ndim == 1:
            return [actions]
        return [actions[i] for i in range(actions.shape[0])]

    # -- Control -----------------------------------------------------------------
    def apply_action(self, action):
        delta_xyz = action[:3]
        delta_euler = action[3:6]
        gripper_cmd = action[-1]

        delta_quat_wxyz = euler2quat(delta_euler[0], delta_euler[1], delta_euler[2], axes="sxyz")
        delta_quat_xyzw = _quat_wxyz_to_xyzw(delta_quat_wxyz)

        delta_affine = Affine(delta_xyz, delta_quat_xyzw)
        motion = CartesianMotion(
            delta_affine,
            reference_type=ReferenceType.Relative,
            relative_dynamics_factor=self.args.relative_dynamics,
            return_when_finished=False,
        )
        # Preemptive, non-blocking update to keep a high-rate control loop
        self.robot.move(motion, asynchronous=True)

        target_width = float(self.gripper.max_width) * float(gripper_cmd)
        if abs(target_width - self.gripper.width) > 0.01:
            self.gripper.move_async(target_width, speed=0.2)

    # -- Main loop ---------------------------------------------------------------
    def run(self):
        step = 0
        try:
            while step < self.args.max_steps:
                obs = self.build_observation()
                if not self.actions_queue:
                    t1 = time.time()
                    actions = self.request_actions(obs)
                    self.actions_queue.extend(actions)
                    print(f"[INFO] Inference time: {time.time() - t1:.3f}s | queued {len(actions)} actions")

                action = self.actions_queue.popleft()
                self.apply_action(action)
                print(f"[STEP {step}] delta_xyz={action[:3]}, delta_euler={action[3:6]}, gripper={action[-1]}")
                step += 1
                time.sleep(self.ctrl_dt)
        except KeyboardInterrupt:
            print("[WARN] Keyboard interrupt; stopping.")
        finally:
            try:
                self.robot.join_motion(timeout=2.0)
            except Exception:
                pass
            self.robot.stop()
            self._shutdown_web_session()

    # -- Web session helpers ----------------------------------------------------
    def _prepare_web_session(self):
        self.web_session = RobotWebSession(self.args.robot_host, self.args.web_username, self.args.web_password)
        try:
            try:
                self.web_session.take_control(wait_timeout=10.0)
            except TakeControlTimeoutError:
                self.web_session.take_control(wait_timeout=30.0, force=True)
            self.web_session.unlock_brakes()
            self.web_session.enable_fci()
        except Exception as exc:
            self.web_session.close()
            raise exc

    def _shutdown_web_session(self):
        if self.web_session is None:
            return
        try:
            self.web_session.disable_fci()
            self.web_session.lock_brakes()
        finally:
            self.web_session.close()


def prepare_record_dir(base_dir: str, prefix: str):
    timestamp = time.strftime(f"{prefix}-%Y-%m-%d-%H-%M-%S")
    record_dir = os.path.join(base_dir, timestamp)
    os.makedirs(record_dir, exist_ok=True)
    return record_dir


def save_run_metadata(record_dir: str, args):
    with open(os.path.join(record_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    os.system(f'git rev-parse HEAD > "{os.path.join(record_dir, "git_commit.txt")}"')
