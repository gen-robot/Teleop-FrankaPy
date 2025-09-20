import os
import cv2
import time
import json
import tyro
import rospy
import threading
import numpy as np
from tqdm import tqdm
from typing import Optional
from termcolor import cprint
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from physics_datagen.ros_toolkit import Ros_listener, run_publisher
from physics_datagen.pyroki_solver import solve_motion, init_pyroki

# Import RealsenseAPI wrapper
from realsense_wrapper import RealsenseAPI
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

@dataclass
class Args:
    """Arguments for object data generation with PyRoKi and RealsenseAPI."""
    
    # Required arguments
    save_dir: str = "/tmp/franka_data"
    """Directory to save the collected data"""
    
    # Optional arguments with defaults
    length: int = 20
    """Number of trajectories to collect"""
    
    cam_width: int = 640
    """Camera width in pixels"""
    
    cam_height: int = 480
    """Camera height in pixels"""
    
    cam_fps: int = 30
    """Camera frame rate"""
    
    urdf: Optional[str] = None
    """Path to robot URDF file (optional, will use default if not provided)"""
    
    debug: int = 0
    """Debug level: 0 (minimal), 1 (normal), 2 (verbose)"""
    
    cam_height: int = 720
    """Camera height in pixels"""
    
    cam_fps: int = 30
    """Camera frames per second"""


stop_event = threading.Event()
K_GAINS = FC.DEFAULT_K_GAINS
D_GAINS = FC.DEFAULT_D_GAINS
CandidateInitPose = [
    [-0.35471786,  0.67136656, -0.12932039, -1.96341863,  0.79877985,  2.14604293,  1.27579102],
    [-1.12162436,  0.99415506,  0.60818567, -1.74134301,  0.3579409,   1.95144463,  1.53270908],
]

def init_realsense_api(height=720, width=1280, fps=30):
    """Initialize RealsenseAPI with specified parameters."""
    camera_api = RealsenseAPI(height=height, width=width, fps=fps, warm_start=60)
    return camera_api

def print_camera_info(camera_api: RealsenseAPI):
    """Print camera information and parameters."""
    num_cameras = camera_api.get_num_cameras()
    cprint(f"Number of cameras detected: {num_cameras}", "cyan")
    
    params = camera_api.get_all_cameras_params()
    for cam_id, cam_params in params.items():
        cprint(f"{cam_id} parameters: {cam_params}", "cyan")

def control_thread(fa, joint_state, joints_traj, init_time, dir_name):
    joints_cmd = []
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    time.sleep(0.2)
    fa.goto_joints(joint_state, duration=5, dynamic=True, buffer_time=10, ignore_virtual_walls=True, 
                     k_gains=K_GAINS,
                     d_gains=D_GAINS,
                   )
    rate = rospy.Rate(20)
    tss = []
    joints_traj.append(joints_traj[-1])
    joints_traj.append(joints_traj[-1])
    joints_traj.append(joints_traj[-1])

    for i in range(0, len(joints_traj)):
        pose = fa.get_pose().translation
        if pose[1]>-0.05:
            flag=1
        else:
            flag=0
        timestamp = rospy.Time.now().to_time() - init_time
        tss.append(timestamp)

        if i == 0:
            cmd = joints_traj[i] 
        else:
            # cmd = joints_traj[len(joints_traj)//2] if flag else joints_traj[-1]
            cmd = joints_traj[i] 
        
        traj_gen_proto_msg = JointPositionSensorMessage(
            id=i, timestamp=rospy.Time.now().to_time() - init_time, 
            joints=cmd,
        )
        joints_cmd.append(cmd)
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
        )
        
        pub.publish(ros_msg)

        rospy.loginfo(f"Published control command ID {traj_gen_proto_msg.id} {cmd}")
        rate.sleep()
    time.sleep(1)
    cmd_timestamps = []
    for i, (ts, cmd) in enumerate(zip(tss, joints_cmd)):
        
        cmd_timestamps.append({
            "id": i,
            "ros_timestamp": ts,
            "cmd": cmd
        })
    with open(f"{dir_name}/control.json", "w") as f:
        json.dump(cmd_timestamps, f, indent=4)

def simple_linear_motion(joint_state, target_joints, num_steps=20):
    """Simple linear interpolation between joint configurations."""
    joint_state = np.array(joint_state)
    target_joints = np.array(target_joints)
    
    trajectory = []
    for i in range(num_steps + 1):
        alpha = i / num_steps
        interp_joints = joint_state + alpha * (target_joints - joint_state)
        trajectory.append(interp_joints.tolist())
    
    return trajectory

def generate_cmd(robot_data, args, joint_state, ee_translation, ee_quaternion, z_proj):
    """Generate command trajectory using PyRoKi instead of CuRobo."""
    robot, robot_coll, world_coll = robot_data
    
    trans_goals = []
    rot_goals = []

    X = 0.05
    for i in range(5):
        trans_goals.append(ee_translation + z_proj * X * i)
        rot_goals.append(ee_quaternion)
    new_joint_state = joint_state
    joints_traj = []
    ee_translation_traj = []
    
    for i in range(0, len(trans_goals)-1):
        result = solve_motion(
            args,
            new_joint_state,
            ee_translation_goal=trans_goals[i+1],
            ee_orientation_goal=rot_goals[i+1],
            robot=robot,
            robot_coll=robot_coll,
            world_coll=world_coll
        )
        
        if result is not None:
            j_traj, ee_traj, new_joint_state = result
            joints_traj += j_traj
            ee_translation_traj += ee_traj
        else:
            cprint(f"Motion planning failed for goal {i+1}", "red")
            break

    joints_traj = joints_traj[::40]
    cprint(f"len of cmd: {len(joints_traj)}", "green")
    return joints_traj

def get_index(args):
    os.makedirs(args.save_dir, exist_ok = True)
    all_entries = os.listdir(args.save_dir)
    all_entries.sort()
    if len(all_entries) >= 1:
        cnt = int(all_entries[-1].split("_")[-1]) + 1
    else: 
        cnt = 0
    return cnt

if __name__ == '__main__':
    args = tyro.cli(Args, description='Pushing with Franka using PyRoKi and RealsenseAPI')

    # Initialize PyRoKi
    robot_data = init_pyroki(args)
    cprint("PyRoKi initialized successfully", "green")
    
    # Initialize camera using RealsenseAPI
    camera_api = init_realsense_api(height=args.cam_height, width=args.cam_width, fps=args.cam_fps)
    cprint(f"Camera initialized with RealsenseAPI: {args.cam_width}x{args.cam_height}@{args.cam_fps}fps", "green")
    
    # Print camera information
    print_camera_info(camera_api)
    
    cnt = get_index(args)
    
    fa = FrankaArm()
    publisher = threading.Thread(target=run_publisher, args=(fa, ))
    publisher.start()
    ros_listener = Ros_listener()
    INIT_POSE = CandidateInitPose[1]
    init_info = {
        "init_pose": INIT_POSE,
        "k_gains": K_GAINS,
        "d_gains": D_GAINS,
    }

    for idx in range(cnt, args.length + 1):
        dir_name = os.path.join(args.save_dir, f"traj_{idx:05d}")
        depth_dir = os.path.join(dir_name, "depth")
        color_dir = os.path.join(dir_name, "rgb")
        vis_dir = os.path.join(dir_name, "vis")   
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        
        with open(os.path.join(dir_name, "init.json"), "w") as f:
            json.dump(init_info, f, indent=4)

        cprint("="*60, "cyan")
        cprint(f"Recording traj {idx} to {dir_name}", "cyan")
        cprint("reset franka", "green")
        try:
            fa.stop_skill()
        except:
            raise EnvironmentError
        fa.goto_joints(INIT_POSE, ignore_virtual_walls=True)
        fa.close_gripper()
        
        joint_state = fa.get_joints().astype('float32')
        ee_translation = fa.get_pose().translation.astype('float32')
        ee_quaternion = fa.get_pose().quaternion.astype('float32')
        rotation = R.from_quat([ee_quaternion[3], ee_quaternion[0], ee_quaternion[1], ee_quaternion[2]])
        rotation_matrix = rotation.as_matrix()
        
        print("ee_translation: ", ee_translation)
        print("ee_quaternion: ", ee_quaternion)
        print("joint_state: ", joint_state)

        z_axis = rotation_matrix[:, 2]
        z_proj = -z_axis
        z_proj[2] = 0.0
        z_proj = z_proj / np.linalg.norm(z_proj)
        print(f"projection of z axis of ee on XoY plain {z_proj}")
        
        joints_traj = generate_cmd(robot_data, args, joint_state, ee_translation, ee_quaternion, z_proj)
        input("Press enter to start moving")
        
        timestamps = []
        
        # Get camera intrinsics using RealsenseAPI
        intrinsics_dict = camera_api.get_intrinsics_dict()
        # Get the first camera's intrinsics (assuming single camera setup)
        # TODO here to control id
        first_camera_id = list(intrinsics_dict.keys())[0]
        color_intrinsics = intrinsics_dict[first_camera_id]
        
        cam_K = np.array([
            [color_intrinsics['fx'], 0, color_intrinsics['ppx']],
            [0, color_intrinsics['fy'], color_intrinsics['ppy']],
            [0, 0, 1]
        ])
        
        # Save camera intrinsics
        with open(os.path.join(dir_name, "cam_K.txt"), "w") as f:
            for row in cam_K:
                f.write(" ".join(f"{x:.10f}" for x in row) + "\n")
        
        # Save camera parameters for reference
        camera_params = camera_api.get_all_cameras_params()
        with open(os.path.join(dir_name, "camera_params.json"), "w") as f:
            json.dump(camera_params, f, indent=4)

        #start controlling
        init_time = rospy.Time.now().to_time()
        control_thread_obj = threading.Thread(target=control_thread, args=(fa, joint_state, joints_traj, init_time, dir_name))
        control_thread_obj.start()
        
        #start recording
        depth_images = []
        color_images = []
        for i in range(90):
            try:
                camera_id = 0
                # Using RealsenseAPI
                rgbd_data = camera_api.get_rgbd()
                # Get the camera_id camera's data (shape: [n_cams, height, width, RGBD])
                if len(rgbd_data.shape) == 4 and rgbd_data.shape[0] > 0:
                    color_image = rgbd_data[camera_id, :, :, :3].astype(np.uint8)  # RGB
                    depth_image = rgbd_data[camera_id, :, :, 3].astype(np.uint16)  # Depth
                    
                    # Convert to the expected format
                    # RealsenseAPI returns RGB format, but we need BGR for opencv
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                    
                    # Camera timestamp (use ROS timestamp as fallback)
                    camera_timestamp = time.time()
                else:
                    cprint("No camera data available!", "red")
                    continue
                
                # Get robot state
                joint_state = ros_listener.joint_state
                ee_trans = ros_listener.ee_pose.translation
                ee_trans = [ee_trans.x, ee_trans.y, ee_trans.z]
                ee_quat = ros_listener.ee_pose.rotation
                ee_quat = [ee_quat.w, ee_quat.x, ee_quat.y, ee_quat.z]
                ros_timestamp = rospy.Time.now().to_time() - init_time

                # Store images
                depth_images.append((depth_image.astype(np.float32) / 1000.0).copy())
                color_images.append(color_image.copy())
                
                timestamps.append({
                        "id": i,
                        "ros_timestamp": ros_timestamp,
                        "camera_timestamp": camera_timestamp,
                        "joint_state": joint_state,
                        "ee_trans": ee_trans,
                        "ee_quat_wxyz": ee_quat
                    })
                    
            except Exception as e:
                cprint(f"Error capturing frame {i}: {e}", "red")
                continue

        control_thread_obj.join()
        for idx, (depth_image, color_image) in enumerate(tqdm(zip(depth_images, color_images), desc='saving...')):
            np.savez_compressed(os.path.join(depth_dir, f"{idx:05d}.npz"), depth=depth_image)
            cv2.imwrite(os.path.join(color_dir, f"{idx:05d}.png"), color_image)

        with open(f"{dir_name}/frame.json", "w") as f:
            json.dump(timestamps, f, indent=4)

    # Clean up camera resources
    camera_api.close()
    cprint("Camera resources cleaned up", "green")

    stop_event.set()
    publisher.join()
