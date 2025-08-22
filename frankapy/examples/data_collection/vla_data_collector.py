import cv2
import os
import h5py
import torch
import time
import imageio
import numpy as np
from PIL import Image
from examples.data_collection.utils.type import get_numpy, to_numpy
from realsense_wrapper.realsense_d435 import RealsenseAPI
from autolab_core import RigidTransform
from frankapy import FrankaArm

class VLADataCollector:
    def __init__(self, robot: FrankaArm, cameras: RealsenseAPI, is_image_encode: bool = False, include_depth: bool = False, store_images_in_npy: bool = False, *args, **kwargs):
        self.robot: FrankaArm = robot
        self.cameras: RealsenseAPI = cameras
        self.is_image_encode = is_image_encode
        self.include_depth = include_depth
        self.store_images_in_npy = store_images_in_npy
        self.data_dict = self.get_empty_data_dict()
        self.device = 'cpu'

    def get_empty_data_dict(self):
        data_dict = {
            "task_info": {
                "instruction": [],
            },
            "action": {
                "end_effector": {
                    "delta_orientation": [],
                    "delta_position": [],
                    "delta_euler": [],
                    "abs_position": [],
                    "abs_euler": [],
                    "gripper_width": [],
                },
                "joint": {
                    "position": [],  # we dont support
                    "gripper_width": [],  # we dont support
                },
            },
            "observation": {
                "is_image_encode": self.is_image_encode,
                "rgb": [],
                "rgb_timestamp": [],
                "depth": [],
                "depth_timestamp": [],
            },
            "state": {
                "end_effector": {
                    "orientation": [],
                    "position": [],
                    "gripper_width": [],
                },
                "joint": {
                    "position": [],
                },
            },
        }
        return data_dict

    def clear_data(self):
        """Clear all collected data."""
        self.data_dict = self.get_empty_data_dict()

    def get_data(self):
        return to_numpy(self.data_dict, self.device)

    def save_multi_cam_videos(self, rgb_array, base_path="videos", fps=30):
        """
        rgb_array shape: [frame_num, cam_num, H, W, 3]
        """
        os.makedirs(base_path, exist_ok=True)

        if isinstance(rgb_array, list):
            rgb_array = np.stack(rgb_array, axis=0)

        if rgb_array.ndim != 5 or rgb_array.shape[-1] != 3:
            raise ValueError("rgb_array must be shape [frame_num, cam_num, H, W, 3]")

        frame_num, cam_num, H, W, _ = rgb_array.shape

        for cam_idx in range(cam_num):
            video_filename = os.path.join(base_path, f"cam_{cam_idx}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1', 'H264'
            out = cv2.VideoWriter(video_filename, fourcc, fps, (W, H))

            for frame_idx in range(frame_num):
                frame = rgb_array[frame_idx, cam_idx]  # [H, W, 3]
                frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"Saved camera {cam_idx} video to: {video_filename}")

    def save_data(self, save_path, episode_idx, is_compressed=False, is_save_video = True):
        """Save data as .npy file with dictionary structure."""
        saving_data = to_numpy(self.data_dict, self.device)

        # Create a copy for npy saving, conditionally excluding image data
        npy_data = saving_data.copy()
        if not self.store_images_in_npy:
            # Remove image data from npy file (new default behavior)
            if "observation" in npy_data:
                npy_data["observation"] = npy_data["observation"].copy()
                npy_data["observation"].pop("rgb", None)
                npy_data["observation"].pop("rgb_timestamp", None)
                npy_data["observation"].pop("depth", None)
                npy_data["observation"].pop("depth_timestamp", None)

        save_func = np.savez_compressed if is_compressed else np.save
        np_path_data = os.path.join(save_path, f"data")
        save_func(np_path_data, npy_data)
        print(f"save data at {np_path_data}.{'npz' if is_compressed else 'npy'}.")

        if is_save_video:
            if self.is_image_encode:
                raise ValueError(f"you set is_image_encode, so the video can not be saved.")
            self.save_multi_cam_videos(saving_data["observation"]["rgb"], save_path)
        self.clear_data()

    def update_instruction(self, instruction):
        self.data_dict["task_info"]["instruction"].append(instruction)

    def update_rgb(self, timestamp=None):
        if self.cameras is None:
            return
        rgb = self.cameras.get_rgb()
        timestamp = time.time() if timestamp==None else timestamp
        if self.is_image_encode:
            success, encoded_rgb = cv2.imencode('.jpeg', get_numpy(rgb, self.device), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise ValueError("JPEG encode error.")
            rgb = np.frombuffer(encoded_rgb.tobytes(), dtype=np.uint8)

        # Always store RGB data for video generation, but conditionally for npy file
        self.data_dict["observation"]["rgb"].append(rgb) # [N, camera_num, 480, 640, 3]
        self.data_dict["observation"]["rgb_timestamp"].append(timestamp)

        if self.include_depth:
            depth = self.cameras.get_depth()
            self.data_dict["observation"]["depth"].append(depth) # [N, camera_num, 480, 640]
            self.data_dict["observation"]["depth_timestamp"].append(timestamp)

    def update_state(self):
        joint_pos = self.robot.get_joints() # [7,]
        pose = self.robot.get_pose()
        gripper_state = self.robot.get_gripper_width()

        self.data_dict["state"]["joint"]["position"].append(joint_pos)
        self.data_dict["state"]["end_effector"]["position"].append(pose.translation)
        self.data_dict["state"]["end_effector"]["orientation"].append(pose.quaternion)
        self.data_dict["state"]["end_effector"]["gripper_width"].append(gripper_state)

    def update_action(self, save_action):
        action = self.data_dict['action']["end_effector"]
        action["delta_position"].append(save_action["delta"]["position"])
        action["delta_orientation"].append(save_action["delta"]["orientation"])
        action["delta_euler"].append(save_action["delta"]["euler_angle"])
        action["abs_position"].append(save_action["abs"]["position"])
        action["abs_euler"].append(save_action["abs"]["euler_angle"])
        action["gripper_width"].append(save_action["gripper_width"])

    def update_data_dict(self, instruction, action, timestamp=None):
        self.update_rgb(timestamp)
        self.update_instruction(instruction)
        self.update_state()
        self.update_action(action)



class VLAHDF5Saver:
    def __init__(self, data_dict, device='cpu'):
        self.data_dict = to_numpy(data_dict, device)

    def save_to_hdf5(self, save_path):
        """Save data as an HDF5 file."""
        with h5py.File(save_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
            # Save task info
            task_info_group = root.create_group("task_info")
            task_info_group.create_dataset("instruction", data=np.array(self.data_dict["task_info"]["instruction"], dtype='S'))

            # Save action data
            action_group = root.create_group("action")
            for key, value in self.data_dict["action"]["end_effector"].items():
                action_group.create_dataset(f"end_effector/{key}", data=np.array(value))
            for key, value in self.data_dict["action"]["joint"].items():
                action_group.create_dataset(f"joint/{key}", data=np.array(value))

            # Save observation data
            observation_group = root.create_group("observation")
            observation_group.create_dataset("is_image_encode", data=self.data_dict["observation"]["is_image_encode"])
            observation_group.create_dataset("rgb_timestamp", data=np.array(self.data_dict["observation"]["rgb_timestamp"]))
            if self.data_dict["observation"]["rgb"]:
                rgb_group = observation_group.create_group("rgb")
                for i, rgb in enumerate(self.data_dict["observation"]["rgb"]):
                    if isinstance(rgb, np.ndarray):
                        rgb_group.create_dataset(str(i), data=rgb)
                    else:
                        rgb_group.create_dataset(str(i), data=np.array(rgb, dtype=np.uint8))

            # Save state data
            state_group = root.create_group("state")
            for key, value in self.data_dict["state"]["end_effector"].items():
                state_group.create_dataset(f"end_effector/{key}", data=np.array(value))
            for key, value in self.data_dict["state"]["joint"].items():
                state_group.create_dataset(f"joint/{key}", data=np.array(value))

        print(f"Data saved to {save_path}.hdf5")
