import os
import cv2
import time
import json
import tyro
import rospy
import yourdfpy
import threading
import transforms3d
import numpy as np
import pyroki as pk
from tqdm import tqdm
from typing import Optional, List
from termcolor import cprint
from dataclasses import dataclass
from physics_datagen.ros_toolkit import Ros_listener, run_publisher

# Import RealsenseAPI wrapper
from realsense_wrapper import RealsenseAPI
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto import JointPositionSensorMessage
from franka_interface_msgs.msg import SensorDataGroup
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from kinematics.solution.solve_ik import solve_batch_ik_with_continuity
from kinematics import PANDA_URDF_PATH


CandidateInitPose = [
    [-0.35471786,  0.67136656, -0.12932039, -1.96341863,  0.79877985,  2.14604293,  1.27579102],
    [-1.12162436,  0.99415506,  0.60818567, -1.74134301,  0.3579409,   1.95144463,  1.53270908],
]

K_GAINS = FC.DEFAULT_K_GAINS
D_GAINS = FC.DEFAULT_D_GAINS


@dataclass
class Args:
    """Arguments for object data generation with PyRoKi and RealsenseAPI."""
    
    # Required arguments
    save_dir: str = ".tmp_data"
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
    
    cam_index: int = 0
    """Camera index (for multi-camera setups)"""
    
    urdf: Optional[str] = PANDA_URDF_PATH
    """Path to robot URDF file (optional, will use default if not provided)"""


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


def visualize(ee_translation):
    """Visualize trajectory points in 3D."""
    import matplotlib.pyplot as plt
    
    # Convert to numpy array if needed
    if isinstance(ee_translation, list):
        ee_translation = np.array(ee_translation)
    
    # Extract x, y, z coordinates
    x = ee_translation[:, 0]
    y = ee_translation[:, 1]
    z = ee_translation[:, 2]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(x, y, z, c='r', marker='o', label='Points')

    # Add axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax.set_title('3D Point Visualization')
    ax.legend()
    
    # Get min/max values for each axis
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    # Calculate maximum range
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Calculate axis centers
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    # Set axis ranges for equal scaling
    ax.set_xlim([x_center - max_range / 2, x_center + max_range / 2])
    ax.set_ylim([y_center - max_range / 2, y_center + max_range / 2])
    ax.set_zlim([z_center - max_range / 2, z_center + max_range / 2])
    plt.show()


def generate_pushing_trajectory(
    robot,
    joint_state: np.ndarray,
    ee_translation: np.ndarray, 
    ee_quaternion: np.ndarray,
    push_direction: np.ndarray,
    push_distance: float = 0.25,
    num_waypoints: int = 5,
    target_link_name: str = "panda_hand_tcp"
) -> List[List[float]]:
    """
    Generate a simple pushing trajectory using batch IK with continuity.
    
    Args:
        robot: Robot object from init_pyroki
        joint_state: Current joint configuration [7-DOF]
        ee_translation: Current end-effector position [x, y, z]
        ee_quaternion: Current end-effector orientation in [w, x, y, z] format
        push_direction: Normalized push direction vector [x, y, z]
        push_distance: Total distance to push (default: 0.25m)
        num_waypoints: Number of waypoints in trajectory
        target_link_name: Name of target link for IK
        
    Returns:
        List of joint configurations forming the trajectory
    """
    
    # Generate waypoints along push direction
    waypoint_positions = []
    waypoint_orientations = []
    
    for i in range(num_waypoints):
        # Linear progression along push direction
        alpha = i / (num_waypoints - 1) if num_waypoints > 1 else 0.0
        position = ee_translation + push_direction * push_distance * alpha
        waypoint_positions.append(position)
        waypoint_orientations.append(ee_quaternion)
    
    # Convert to arrays
    waypoint_positions = np.array(waypoint_positions)
    waypoint_orientations = np.array(waypoint_orientations)
    
    try:
        # Solve batch IK with continuity
        joint_solutions = solve_batch_ik_with_continuity(
            robot=robot,
            target_link_name=target_link_name,
            target_wxyz_sequence=waypoint_orientations,
            target_position_sequence=waypoint_positions,
            initial_guess=joint_state
        )
        
        # Convert to list format
        trajectory = []
        for joint_config in joint_solutions:
            trajectory.append(joint_config.tolist())
            
        cprint(f"Generated {len(trajectory)} waypoints using batch IK", "green")
        return trajectory
        
    except Exception as e:
        cprint(f"Batch IK failed: {e}, using fallback linear interpolation", "yellow")
        raise ValueError("Batch IK failed") from e


def generate_cmd(robot, joint_state, ee_translation, ee_quaternion, z_proj):
    """Generate command trajectory using simplified pushing pattern generator."""
    try:
        # Use the new simplified trajectory generator
        joints_traj = generate_pushing_trajectory(
            robot=robot,
            joint_state=joint_state,
            ee_translation=ee_translation,
            ee_quaternion=ee_quaternion,
            push_direction=z_proj,
            num_waypoints=1000,
            push_distance=0.1,
        )
        
        cprint(f"Generated trajectory with {len(joints_traj)} waypoints", "green")
        return joints_traj
        
    except Exception as e:
        cprint(f"Trajectory generation failed: {e}, using fallback", "red")
        # Fallback: just return current position repeated
        return [joint_state.tolist()] * 20


def get_index(args):
    os.makedirs(args.save_dir, exist_ok = True)
    all_entries = os.listdir(args.save_dir)
    all_entries.sort()
    if len(all_entries) >= 1:
        index = int(all_entries[-1].split("_")[-1]) + 1
    else: 
        index = 0
    return index


if __name__ == '__main__':
    
    args = tyro.cli(Args, description='Pushing with Franka using PyRoKi and RealsenseAPI')
    urdf = yourdfpy.URDF.load(args.urdf)
    robot = pk.Robot.from_urdf(urdf)    
    # Initialize camera using RealsenseAPI
    camera_api = init_realsense_api(height=args.cam_height, width=args.cam_width, fps=args.cam_fps)
    cprint(f"Camera initialized with RealsenseAPI: {args.cam_width}x{args.cam_height}@{args.cam_fps}fps", "green")
    print_camera_info(camera_api)
    fa = FrankaArm()
    publisher = threading.Thread(target=run_publisher, args=(fa,))
    publisher.start()
    ros_listener = Ros_listener()
    
    # log
    init_info = {
        "init_pose": CandidateInitPose[1],
        "k_gains": K_GAINS,
        "d_gains": D_GAINS,
    }
    
    init_cnt = get_index(args)
    for idx in range(init_cnt, args.length + 1):
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

        fa.goto_joints(init_info['init_pose'], ignore_virtual_walls=True)
        fa.close_gripper()
        
        joint_state = fa.get_joints().astype('float32')
        ee_translation = fa.get_pose().translation.astype('float32')
        ee_quaternion = fa.get_pose().quaternion.astype('float32')
        rotation_matrix = transforms3d.quaternions.quat2mat(ee_quaternion) # need to be quat_wxyz
        
        print("ee_translation: ", ee_translation)
        print("ee_quaternion: ", ee_quaternion)
        print("joint_state: ", joint_state)

        z_axis = rotation_matrix[:, 2]
        z_proj = -z_axis
        z_proj[2] = 0.0
        z_proj = z_proj / np.linalg.norm(z_proj)
        print(f"projection of z axis of ee on XoY plain {z_proj}")
        
        joints_traj = generate_cmd(robot, joint_state, ee_translation, ee_quaternion, z_proj)
        input("Press enter to start moving")
        
        timestamps = []
        
        # Get camera intrinsics using RealsenseAPI
        intrinsics_dict = camera_api.get_intrinsics_dict()
        # Get the first camera's intrinsics (assuming single camera setup)
        color_intrinsics = intrinsics_dict[list(intrinsics_dict.keys())[args.cam_index]]
        
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
                # Using RealsenseAPI
                rgbd_data = camera_api.get_rgbd()
                # Get the camera_id camera's data (shape: [n_cams, height, width, RGBD])
                if len(rgbd_data.shape) == 4 and rgbd_data.shape[0] > 0:
                    color_image = rgbd_data[args.cam_index, :, :, :3].astype(np.uint8)  # RGB
                    depth_image = rgbd_data[args.cam_index, :, :, 3].astype(np.uint16)  # Depth
                    
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

    publisher.join()
