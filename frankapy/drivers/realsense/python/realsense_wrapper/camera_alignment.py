import numpy as np
import pyrealsense2 as rs
from collections import OrderedDict

import time
from PIL import Image
import cv2


class RealsenseAPI:
    """Wrapper that implements boilerplate code for RealSense cameras"""

    def __init__(self, height=480, width=640, fps=30, warm_start=60):
        self.height = height
        self.width = width
        self.fps = fps

        # Identify devices
        self.device_ls = []
        for c in rs.context().query_devices():
            self.device_ls.append(c.get_info(rs.camera_info(1)))

        # Start stream
        print(f"Connecting to RealSense cameras ({len(self.device_ls)} found) ...")
        self.pipes = []
        self.profiles = OrderedDict()
        for i, device_id in enumerate(self.device_ls):
            pipe = rs.pipeline()
            config = rs.config()

            config.enable_device(device_id)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
            )

            self.pipes.append(pipe)
            self.profiles[device_id]=pipe.start(config)

            print(f"Connected to camera {i+1} ({device_id}).")

        self.align = rs.align(rs.stream.color)
        # Warm start camera (realsense automatically adjusts brightness during initial frames)
        for _ in range(warm_start):
            self._get_frames()

    def _get_frames(self):
        framesets = [pipe.wait_for_frames() for pipe in self.pipes]
        return [self.align.process(frameset) for frameset in framesets]

    def get_intrinsics(self):
        intrinsics_ls = []
        for profile in self.profiles.values():
            stream = profile.get_streams()[1]
            intrinsics = stream.as_video_stream_profile().get_intrinsics()

            intrinsics_ls.append(intrinsics)

        return intrinsics_ls

    def get_intrinsics_dict(self):
        intrinsics_ls = OrderedDict()
        for device_id, profile in self.profiles.items():
            stream = profile.get_streams()[1]
            intrinsics = stream.as_video_stream_profile().get_intrinsics()
            param_dict = dict([(p, getattr(intrinsics, p)) for p in dir(intrinsics) if not p.startswith('__')])
            param_dict['model'] = param_dict['model'].name

            intrinsics_ls[device_id] = param_dict

        return intrinsics_ls
    
    def get_num_cameras(self):
        return len(self.device_ls)

    def get_rgbd(self):
        """Returns a numpy array of [n_cams, height, width, RGBD]"""
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()

        rgbd = np.empty([num_cams, self.height, self.width, 4], dtype=np.uint16)

        for i, frameset in enumerate(framesets):
            color_frame = frameset.get_color_frame()
            rgbd[i, :, :, :3] = np.asanyarray(color_frame.get_data())

            depth_frame = frameset.get_depth_frame()
            rgbd[i, :, :, 3] = np.asanyarray(depth_frame.get_data())

        return rgbd


    def get_rgb(self):
        """Returns a numpy array of [n_cams, height, width, RGB]"""
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()

        rgb = np.empty([num_cams, self.height, self.width, 3], dtype=np.uint8)

        for i, frameset in enumerate(framesets):
            color_frame = frameset.get_color_frame()
            rgb[i, :, :, :] = np.asanyarray(color_frame.get_data())

        # print("check:", rgb.shape)

        return rgb

    def get_depth(self):
        """Returns a numpy array of [n_cams, height, width, depth]"""
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()

        depth = np.empty([num_cams, self.height, self.width], dtype=np.uint16)

        for i, frameset in enumerate(framesets):
            depth_frame = frameset.get_depth_frame()
            depth[i, :, :] = np.asanyarray(depth_frame.get_data())

        return depth

def render(image: np.ndarray, window_name="Overlay Viewer"):
    cv2.imshow(window_name, image[:, :, ::-1])  # 将RGB转换为BGR供OpenCV显示
    cv2.waitKey(1)

if __name__ == "__main__":
    cams = RealsenseAPI()
    print(f"Num cameras: {cams.get_num_cameras()}")

    # 读取参考图像并转换为numpy格式
    ref_image = Image.open("camera_ref.jpeg").convert("RGB")
    ref_image = np.array(ref_image).astype(np.uint8)

    if ref_image.shape != (480, 480, 3):
        raise ValueError(f"camera_ref.png must be 480x480x3, but got {ref_image.shape}")

    alpha = 0.5  # 透明度参数，可调整

    while True:
        # 获取图像
        rgb = cams.get_rgb()
        # 裁剪rgb[1]：从(640,480)裁为(480,480)，方式为水平截取40:520
        cropped = rgb[1][:, 40:520, :].astype(np.uint8)

        # 叠加两张图
        overlay = cv2.addWeighted(cropped, alpha, ref_image, 1 - alpha, 0)

        # 显示
        render(overlay)

        # 控制帧率为20Hz
        time.sleep(1 / 20)
