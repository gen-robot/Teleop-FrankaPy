"""
VLA deployment script with IK-based joint control.
This script uses IK to convert EE pose commands to joint commands,
aligning the low-level control interface between real robot and simulation.

Reference: examples/data_collection/data_collection_with_ik.py
"""
import os
import sys
import time
import json
import torch
import cv2
import imageio
import rospy
import requests
import argparse
import json_numpy
import numpy as np
from pathlib import Path
from collections import deque
from PIL import Image as PImage
from transforms3d.euler import euler2quat, euler2mat, quat2euler, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from realsense_wrapper.realsense_d435 import RealsenseAPI
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.franka_constants import FrankaConstants as FC
from frankapy.proto import JointPositionSensorMessage, PosePositionSensorMessage, CartesianImpedanceSensorMessage

# IK solver imports
from kinematics import PANDA_URDF_PATH
from kinematics.panda_ik_solver_sim_aligned import create_sim_aligned_ik_solver, SimPose

K_GAINS = [400.0, 250.0, 400.0, 300.0, 350.0, 150.0, 80.0]
D_GAINS = [100.0, 45.0, 80.0, 45.0, 40.0, 22.0, 15.0]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instructions', type=str, default="test")
    parser.add_argument('--ctrl_freq', type=float, default=5.0)
    parser.add_argument('--record_dir', type=str, default='logs/openpi')
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--vla_server_ip', type=str, default='localhost', help='The IP address of the VLA server')
    parser.add_argument('--vla_server_port', type=int, default=9876, help='The port of the VLA server')
    parser.add_argument('--episode_idx', type=str, required=True)
    parser.add_argument('--chunk_size', type=int, default=16)
    # IK-related arguments
    parser.add_argument('--use_gpu_ik', action='store_true', help='Use GPU for IK solver')
    parser.add_argument('--verify_ik', action='store_true', help='Verify IK solution with FK (for debugging)')

    return parser.parse_args()


class VLADeployWithIK:
    """VLA deployment class that uses IK to convert EE pose to joint positions.
    
    Uses SimAlignedPandaIKSolver - fully aligned with ManiSkill simulation kinematics.
    """
    
    def __init__(self, args):
        self.args = args
        self.observation_window = deque(maxlen=2)

        # Interfaces
        self.robot = FrankaArm()
        self.camera = RealsenseAPI()

        # Record settings
        self.record_dir = args.record_dir
        self.episode_idx = args.episode_idx
        self.chunk_size = args.chunk_size
        os.makedirs(self.record_dir, exist_ok=True)

        self.init_xyz = None
        self.init_rotation = None
        self.command_xyz = None
        self.command_rotation = None
        self.current_joints = None  # For IK warm start
        self.actions_list = []
        self.actions_record_list = []

        self.max_steps = args.max_steps
        self.ctrl_freq = args.ctrl_freq
        self.act_url = f"http://{args.vla_server_ip}:{args.vla_server_port}/act"
        self.init_time = rospy.Time.now().to_time()
        self.record_image = []
        
        # IK statistics
        self.ik_success_count = 0
        self.ik_failure_count = 0

        # Initialize IK solver
        self._init_ik_solver()

    def _init_ik_solver(self):
        """Initialize the simulation-aligned Panda IK solver."""
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

    # maybe can be used for aligning with training
    def _jpeg_mapping(self, img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    def update_observation_window(self):
        images = self.camera.get_rgb()
        mix_image = np.concatenate([images[1], images[0]], axis=1)
        self.record_image.append(mix_image)
        self.observation_window.append({
            'ee_pose_T': self.robot.get_pose().matrix,  # np shape (4,4)
            'joints': self.robot.get_joints(),  # np shape (7,)
            'gripper_width': np.array([self.robot.get_gripper_width()]),  # np shape(1,)
            'instruction': self.args.instructions,  # string
            'images': images.astype(np.uint8)  # support multi camera
        })

    def ee_pose_init(self):
        """Initialize the end-effector pose and joint positions."""
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

    def robot_init(self):
        self.robot.reset_joints()
        self.robot.open_gripper()
        input("[INFO] Press enter to continue")
        
        # Get home joints for dynamic joint control
        home_joints = self.robot.get_joints()
        print(f"[INFO] Home joints: {home_joints}")
        
        # Start dynamic joint control mode (instead of pose control)
        self.robot.goto_joints(
            home_joints, 
            duration=10, 
            dynamic=True, 
            buffer_time=100000000, 
            skill_desc='JOINT_CONTROL_IK',
            k_gains=K_GAINS,
            d_gains=D_GAINS,
        )

    def run_inference_loop(self):
        """Main inference loop using IK-based joint control."""
        step = 0
        self.ee_pose_init()
        control_rate = rospy.Rate(self.ctrl_freq)
        print("[INFO] Starting inference loop with IK-based joint control...")
        
        try:
            while step < self.max_steps:
                self.update_observation_window()
                observation = self.observation_window[-1]

                if len(self.actions_list) == 0:
                    # request and inference
                    t1 = time.time()

                    payload = {
                        "ee_pose_T": observation['ee_pose_T'],
                        "joints": observation['joints'],
                        "gripper_width": observation['gripper_width'],
                        "images": observation['images'], 
                        "instruction": observation['instruction'],
                    }
                    payload_string = json_numpy.dumps(payload)

                    response = requests.post(
                        self.act_url,
                        data=payload_string,
                        headers={"Content-Type": "application/json"},
                        timeout=1000,
                    )

                    if response.status_code == 200:
                        response_data = json_numpy.loads(response.text)
                        action = np.array(response_data['actions'])
                    else:
                        raise TimeoutError(" >>>>> Read response timeout <<<<< ")
                    
                    if len(action.shape) == 1:
                        self.actions_list.append(action)
                    else:
                        for idx in range(min(action.shape[0], self.chunk_size)):
                            self.actions_list.append(action[idx])
                
                action = self.actions_list.pop(0)
                print("request and inference time cost", time.time() - t1, "| action.shape", action.shape)

                timestamp = rospy.Time.now().to_time() - self.init_time
                delta_xyz, delta_euler, gripper = action[:3], action[3:6], action[-1]
                delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz')
                
                # Compute target pose
                self.command_xyz = self.command_xyz + delta_xyz
                self.command_rotation = np.matmul(self.command_rotation, delta_rotation)

                try:
                    # Compute IK to get target joint positions
                    target_joints = self._compute_ik_for_pose(
                        self.command_xyz,
                        self.command_rotation,
                        initial_joints=self.current_joints
                    )

                    if target_joints is None:
                        print(f"[WARNING] IK failed at step {step}, skipping this command")
                        self.ik_failure_count += 1
                        control_rate.sleep()
                        continue
                    
                    self.ik_success_count += 1
                    
                    # Verify IK solution (optional, for debugging)
                    if self.args.verify_ik and step % 10 == 0:
                        quat_wxyz = mat2quat(self.command_rotation)
                        target_pose_for_verify = SimPose.from_pq(
                            position=self.command_xyz,
                            quaternion=quat_wxyz,
                            device=self.ik_solver.device
                        )
                        pos_error, ori_error = self.ik_solver.verify_ik_solution(
                            torch.tensor([target_joints], device=self.ik_solver.device),
                            target_pose_for_verify
                        )
                        print(f"[DEBUG] Step {step}: "
                              f"pos_err={pos_error*1000:.3f}mm, "
                              f"ori_err={np.rad2deg(ori_error):.3f}deg")

                    gripper_width = FC.GRIPPER_WIDTH_MAX * gripper

                    # Send joint command to robot using dynamic joint control
                    joint_proto_msg = JointPositionSensorMessage(
                        id=step + 1,
                        timestamp=timestamp,
                        joints=target_joints.tolist()
                    )
                    
                    # Use joint position sensor message
                    ros_pub_sensor_msg = make_sensor_group_msg(
                        trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                            joint_proto_msg, SensorDataMessageType.JOINT_POSITION
                        )
                    )
                    
                    rospy.loginfo(f'Publishing: Step {step+1}, IK joints sent')
                    self.robot.publish_sensor_values(ros_pub_sensor_msg)
                    
                    # Update current joints for next IK warm start
                    self.current_joints = target_joints

                    # Control gripper
                    current_gripper_width = self.robot.get_gripper_width()
                    if abs(gripper_width - current_gripper_width) > 0.01:
                        grasp = True if gripper < 0.5 else False
                        gripper_width = np.clip(gripper_width, 0.015, 0.07)
                        self.robot.goto_gripper(
                            gripper_width, 
                            epsilon_inner=0.06, 
                            epsilon_outer=0.06, 
                            grasp=grasp, 
                            force=FC.GRIPPER_MAX_FORCE / 3.0, 
                            speed=0.12, 
                            block=True, 
                            skill_desc="control_gripper"
                        )
                        
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        self.robot.stop_skill()
                        print(f"[WARN] Keyboard Interrupt: {e}")
                        break
                    self.ee_pose_init()
                    control_rate.sleep()
                    print(f"[WARN] Move failed?: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                print(f"[STEP {step}] delta_xyz: {delta_xyz}, delta_euler: {delta_euler}, gripper: {gripper}")
                step_record = {
                    "step": step,
                    "delta_xyz": delta_xyz.tolist(),
                    "delta_euler": delta_euler.tolist(),
                    "gripper": gripper.tolist() if hasattr(gripper, 'tolist') else float(gripper),
                    "target_joints": target_joints.tolist(),  # Also record joint positions
                }
                self.actions_record_list.append(step_record)
                step += 1
                control_rate.sleep()
                
        except Exception as e:
            self.robot.stop_skill()
            control_rate.sleep()
            print(f"[WARN] Exception: {e}")
            import traceback
            traceback.print_exc()

        # Print IK statistics
        total_ik = self.ik_success_count + self.ik_failure_count
        if total_ik > 0:
            success_rate = 100.0 * self.ik_success_count / total_ik
            print(f"\n[INFO] IK Statistics:")
            print(f"  Success: {self.ik_success_count}")
            print(f"  Failure: {self.ik_failure_count}")
            print(f"  Success Rate: {success_rate:.2f}%")

        self.robot.stop_skill()
        rospy.loginfo('Done')
        
        # Save video and action logs
        video_logpath = os.path.join(self.record_dir, f"log_policy_deploy_ik.mp4")
        print(video_logpath)
        imageio.mimsave(video_logpath, self.record_image, fps=self.ctrl_freq)
        
        actions_logpath = os.path.join(self.record_dir, f"log_policy_output_ik.json")
        with open(actions_logpath, 'w') as f:
            json.dump(self.actions_record_list, f, indent=4)
        
        # Save IK statistics
        ik_stats = {
            "ik_success_count": self.ik_success_count,
            "ik_failure_count": self.ik_failure_count,
            "ik_success_rate": success_rate if total_ik > 0 else 0,
            "ik_solver_mode": self.ik_solver.alignment_mode,
            "ik_max_iterations": self.ik_solver.max_iterations,
        }
        ik_stats_path = os.path.join(self.record_dir, f"ik_statistics.json")
        with open(ik_stats_path, 'w') as f:
            json.dump(ik_stats, f, indent=4)
            
        print("[INFO] Reaching Max-steps, Inference loop finished.")


def main():
    args = parse_arguments()
    args.record_dir = os.path.join(args.record_dir, args.episode_idx)
    os.makedirs(args.record_dir, exist_ok=True)

    # Save arguments and git commit
    with open(os.path.join(args.record_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    os.system(f'git rev-parse HEAD > {os.path.join(args.record_dir, "git_commit.txt")}')

    agent = VLADeployWithIK(args)
    agent.robot_init()
    agent.run_inference_loop()


if __name__ == '__main__':
    main()
