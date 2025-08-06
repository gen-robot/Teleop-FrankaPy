import os
import sys
import time
import json
import torch
import cv2
import rospy
import requests
import tyro
import json_numpy
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from PIL import Image as PImage
from transforms3d.euler import euler2quat, euler2mat, quat2euler
from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from realsense_wrapper.realsense_d435 import RealsenseAPI
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.franka_constants import FrankaConstants as FC
from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage
from query_utils import SimpleLogger, VLAClient, GracefulExit, safe_execute
json_numpy.patch()

@dataclass
class Args:
    """Parameters for VLA deployment and robot control."""

    instructions: str = "test"
    """Task instruction for the VLA model."""

    ctrl_freq: float = 5.0
    """Control frequency in Hz."""

    record_dir: str = "logs/vla"
    """Directory path for recording data and videos."""

    max_steps: int = 500
    """Maximum number of steps to execute."""

    vla_server_ip: str = "localhost"
    """IP address of the VLA server."""

    vla_server_port: int = 9876
    """Port number of the VLA server."""

    enable_logging: bool = False
    """Whether to enable data logging."""

    recording_video_fps: int = 10
    """Video recording frame rate."""

class VLADeploy:
    def __init__(self, args: Args):
        self.args = args
        self.observation_window = deque(maxlen=2)

        # Initialize robot and camera interfaces
        self.robot = FrankaArm()
        self.camera = RealsenseAPI()

        # Recording setup
        self.record_dir = args.record_dir
        os.makedirs(self.record_dir, exist_ok=True)

        self.init_xyz = None
        self.init_rotation = None
        self.command_xyz = None
        self.command_rotation = None
        self.actions_list = []

        self.max_steps = args.max_steps
        self.ctrl_freq = args.ctrl_freq
        self.init_time = rospy.Time.now().to_time()
        
        self.logger = SimpleLogger(
            record_dir=self.record_dir,
            enable_logging=args.enable_logging,
            recording_video_fps=args.recording_video_fps
        )
        
        # VLA client for action inference
        self.vla_client = VLAClient(
            server_ip=args.vla_server_ip,
            server_port=args.vla_server_port,
            timeout=10,
            max_retries=3
        )
        
        self.exit_handler = GracefulExit()
        self.exit_handler.add_cleanup(self._cleanup)

    def _cleanup(self):
        try:
            self.robot.stop_skill()
        except:
            print("stop skill failed")
        self.logger.save_and_cleanup()

    def _jpeg_mapping(self, img):
        """JPEG compression/decompression for training alignment"""
        img = cv2.imencode('.jpg', img)[1].tobytes()
        return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    def update_observation_window(self):
        images = self.camera.get_rgb()
        observation = {
            'ee_pose_T': self.robot.get_pose().matrix,  # Shape (4,4)
            'joints': self.robot.get_joints(),  # Shape (7,)
            'gripper_width': np.array([self.robot.get_gripper_width()]),  # Shape (1,)
            'instruction': self.args.instructions,
            'images': images.astype(np.uint8)  # Multi-camera support
        }
        self.observation_window.append(observation)
        return observation

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
        self.robot.goto_pose(FC.HOME_POSE, duration=10, dynamic=True, buffer_time=100000000, skill_desc='MOVE', 
                        cartesian_impedances=FC.DEFAULT_CARTESIAN_IMPEDANCES, ignore_virtual_walls=True)

    @safe_execute
    def execute_robot_action(self, step, action, timestamp):
        """Execute robot action with delta pose and gripper command"""
        delta_xyz, delta_euler, gripper = action[:3], action[3:6], np.clip(action[-1], 0, 1)
        delta_rotation = euler2mat(delta_euler[0], delta_euler[1], delta_euler[2], 'sxyz')
        
        # Calculate target pose
        self.command_xyz += delta_xyz
        self.command_rotation = np.matmul(self.command_rotation, delta_rotation)
        
        self.command_transform = RigidTransform(
            rotation=self.command_rotation, 
            translation=self.command_xyz, 
            from_frame='franka_tool', 
            to_frame='world'
        )
        gripper_width = FC.GRIPPER_WIDTH_MAX * gripper

        # Create ROS control messages
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
        
        # Uncomment to actually control the robot
        rospy.loginfo(f'Publishing: Steps {step+1}, delta_xyz = {delta_xyz}')
        self.robot.publish_sensor_values(ros_pub_sensor_msg)

        current_gripper_width = self.robot.get_gripper_width()
        if abs(gripper_width - current_gripper_width) > 0.01:
            grasp = True if gripper < 0.5 else False
            # Uncomment to control gripper
            self.robot.goto_gripper(gripper_width, grasp=grasp, force=FC.GRIPPER_MAX_FORCE/3.0, speed=0.12, block=False, skill_desc="control_gripper")

        return True

    def run_inference_loop(self):
        step = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        self.ee_pose_init()
        control_rate = rospy.Rate(self.ctrl_freq)
        print("[INFO] Starting inference loop...")
        
        try:
            while step < self.max_steps and not self.exit_handler.should_exit():

                observation = self.update_observation_window()

                # Request new actions when buffer is empty
                if len(self.actions_list) == 0:
                    print(f"[STEP {step}] Requesting action...")
                    
                    success, response_data, inference_time = self.vla_client.send_request({
                        "ee_pose_T": observation['ee_pose_T'],
                        "joints": observation['joints'],
                        "gripper_width": observation['gripper_width'],
                        "images": observation['images'], 
                        "instruction": observation['instruction'],
                    })
                    
                    if success:
                        action = np.array(response_data['actions'])
                        if len(action.shape) == 1:
                            self.actions_list.append(action)
                        else:
                            for idx in range(action.shape[0]):
                                self.actions_list.append(action[idx])
                                
                        consecutive_failures = 0
                        print(f"[STEP {step}] Action received ({inference_time:.3f}s)")
                        
                    else:
                        consecutive_failures += 1
                        print(f"[STEP {step}] Request failed ({consecutive_failures}/{max_consecutive_failures})")
                        
                        if consecutive_failures >= max_consecutive_failures:
                            print("[INFO] Server disconnected, stopping...")
                            break
                            
                        control_rate.sleep()
                        continue

                # Execute buffered actions
                if len(self.actions_list) > 0:
                    action = self.actions_list.pop(0)
                    timestamp = rospy.Time.now().to_time() - self.init_time
                    
                    if self.execute_robot_action(step, action, timestamp):
                        self.logger.log_step_data(
                            step, 
                            observation, 
                            action_executed=action,
                            command_xyz=self.command_xyz, 
                            command_rotation=self.command_rotation
                        )
                        print(f"[STEP {step}] Executed action xyz: {action[:3]} | gripper: {action[-1]:.3f}")
                        step += 1
                    else:
                        print(f"[STEP {step}] Execution failed, reinitializing...")
                        self.ee_pose_init()
                
                control_rate.sleep()
                
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
        finally:
            self.robot.stop_skill()
            print(f"[INFO] Completed {step} steps")

def main():
    args = tyro.cli(Args)
    
    # Create timestamped directory
    timestamp = time.strftime("VLA-%Y-%m-%d-%H-%M-%S")
    args.record_dir = os.path.join(args.record_dir, timestamp)
    os.makedirs(args.record_dir, exist_ok=True)

    # Save configuration and git commit
    with open(os.path.join(args.record_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    os.system(f'git rev-parse HEAD > {os.path.join(args.record_dir, "git_commit.txt")}')

    agent = VLADeploy(args)
    agent.robot_init()
    agent.run_inference_loop()

if __name__ == '__main__':
    main()


