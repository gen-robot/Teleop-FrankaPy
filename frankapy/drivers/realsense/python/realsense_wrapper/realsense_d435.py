import numpy as np
import pyrealsense2 as rs
from collections import OrderedDict
import os
import argparse

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
        self.sensors = OrderedDict()  # Store sensors for parameter control
        
        for i, device_id in enumerate(self.device_ls):
            pipe = rs.pipeline()
            config = rs.config()

            config.enable_device(device_id)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
            )

            self.pipes.append(pipe)
            self.profiles[device_id] = pipe.start(config)
            
            # Get device and sensors for parameter control
            device = rs.context().query_devices()[i]
            color_sensor = device.first_color_sensor()
            depth_sensor = device.first_depth_sensor()
            self.sensors[device_id] = {
                'color': color_sensor,
                'depth': depth_sensor,
                'device': device
            }

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

    def get_key_camera_params(self, device_index=0):
        """Get key camera parameters for filename generation."""
        try:
            device_id = self.device_ls[device_index]
            color_sensor = self.sensors[device_id]['color']
            depth_sensor = self.sensors[device_id]['depth']
            
            params = {}
            
            # Color sensor parameters
            try:
                if color_sensor.supports(rs.option.exposure):
                    params['exp'] = int(color_sensor.get_option(rs.option.exposure))
                if color_sensor.supports(rs.option.gain):
                    params['gain'] = int(color_sensor.get_option(rs.option.gain))
                if color_sensor.supports(rs.option.white_balance):
                    params['wb'] = int(color_sensor.get_option(rs.option.white_balance))
                
                # Check auto exposure status
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    params['auto_exp'] = bool(color_sensor.get_option(rs.option.enable_auto_exposure))
                
                # Check auto white balance status
                if color_sensor.supports(rs.option.enable_auto_white_balance):
                    params['auto_wb'] = bool(color_sensor.get_option(rs.option.enable_auto_white_balance))
                    
            except Exception:
                pass
            
            # Depth sensor parameters
            try:
                if depth_sensor.supports(rs.option.laser_power):
                    params['laser'] = int(depth_sensor.get_option(rs.option.laser_power))
            except Exception:
                pass
            
            return params
            
        except Exception as e:
            print(f"Failed to get camera parameters for camera {device_index+1}: {e}")
            return {}

    def get_all_cameras_params(self):
        """Get key parameters for all cameras."""
        all_params = {}
        for i in range(self.get_num_cameras()):
            all_params[f'cam{i+1}'] = self.get_key_camera_params(i)
        return all_params

    # New camera parameter control methods
    def set_exposure(self, device_index=0, exposure_value=None):
        """Set exposure for color sensor. If None, enables auto exposure."""
        try:
            device_id = self.device_ls[device_index]
            color_sensor = self.sensors[device_id]['color']
            
            if exposure_value is None:
                # Enable auto exposure
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                print(f"Camera {device_index+1}: Auto exposure enabled")
            else:
                # Disable auto exposure and set manual value
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, exposure_value)
                print(f"Camera {device_index+1}: Manual exposure set to {exposure_value}")
        except Exception as e:
            print(f"Failed to set exposure for camera {device_index+1}: {e}")

    def set_gain(self, device_index=0, gain_value=None):
        """Set gain for color sensor. If None, enables auto gain."""
        try:
            device_id = self.device_ls[device_index]
            color_sensor = self.sensors[device_id]['color']
            
            if gain_value is None:
                # Enable auto gain
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                print(f"Camera {device_index+1}: Auto gain enabled")
            else:
                # Set manual gain
                color_sensor.set_option(rs.option.gain, gain_value)
                print(f"Camera {device_index+1}: Gain set to {gain_value}")
        except Exception as e:
            print(f"Failed to set gain for camera {device_index+1}: {e}")

    def set_laser_power(self, device_index=0, power_value=150):
        """Set laser power for depth sensor (0-360)."""
        try:
            device_id = self.device_ls[device_index]
            depth_sensor = self.sensors[device_id]['depth']
            
            depth_sensor.set_option(rs.option.laser_power, power_value)
            print(f"Camera {device_index+1}: Laser power set to {power_value}")
        except Exception as e:
            print(f"Failed to set laser power for camera {device_index+1}: {e}")

    def set_white_balance(self, device_index=0, wb_value=None):
        """Set white balance for color sensor. If None, enables auto white balance."""
        try:
            device_id = self.device_ls[device_index]
            color_sensor = self.sensors[device_id]['color']
            
            if wb_value is None:
                # Enable auto white balance
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
                print(f"Camera {device_index+1}: Auto white balance enabled")
            else:
                # Disable auto and set manual value
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
                color_sensor.set_option(rs.option.white_balance, wb_value)
                print(f"Camera {device_index+1}: White balance set to {wb_value}")
        except Exception as e:
            print(f"Failed to set white balance for camera {device_index+1}: {e}")

    def get_camera_options(self, device_index=0):
        """Get available options and their current values for a camera."""
        try:
            device_id = self.device_ls[device_index]
            sensors = self.sensors[device_id]
            
            options_info = {}
            for sensor_type, sensor in sensors.items():
                if sensor_type == 'device':
                    continue
                options_info[sensor_type] = {}
                for option in sensor.get_supported_options():
                    try:
                        if not sensor.is_option_read_only(option):
                            current_value = sensor.get_option(option)
                            option_range = sensor.get_option_range(option)
                            options_info[sensor_type][option.name] = {
                                'current': current_value,
                                'min': option_range.min,
                                'max': option_range.max,
                                'step': option_range.step,
                                'default': option_range.default
                            }
                    except Exception:
                        continue
            
            return options_info
        except Exception as e:
            print(f"Failed to get options for camera {device_index+1}: {e}")
            return {}

    def close(self):
        """Properly close all camera streams."""
        for pipe in self.pipes:
            pipe.stop()
        print("All camera streams closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx',type=str,required=True)
    args = parser.parse_args()
    cams = RealsenseAPI()

    print(f"Num cameras: {cams.get_num_cameras()}")
    
    # # Test camera options
    # if cams.get_num_cameras() > 0:
    #     options = cams.get_camera_options(0)
    #     print("Camera 0 options:", options)
        
    #     # Test key parameters
    #     key_params = cams.get_key_camera_params(0)
    #     print("Camera 0 key params:", key_params)
    
    intrinsics_list = cams.get_intrinsics()
    print("\n[1] Intrinsics list returned by get_intrinsics() (rs.intrinsics objects):")
    print(intrinsics_list)
    print("\n[2] Detailed parameters for each camera:")
    for cam_idx, intrinsics in enumerate(intrinsics_list):
        print(f"\nIntrinsics for camera {cam_idx + 1}")
        print(f"Resolution (width x height): {intrinsics.width}x{intrinsics.height}")
        print(f"Focal length fx: {intrinsics.fx:.6f}")
        print(f"Focal length fy: {intrinsics.fy:.6f}")
        print(f"Principal point ppx: {intrinsics.ppx:.6f}")
        print(f"Principal point ppy: {intrinsics.ppy:.6f}")
        print(f"Distortion model: {intrinsics.model.name}")
        print(f"Distortion coefficients: {[round(c,6) for c in intrinsics.coeffs]}")

    rgbd = cams.get_rgbd()
    rgb = cams.get_rgb()
    from PIL import Image
    image = Image.fromarray(rgb[0].astype(np.uint8)) # ,"RGB"
    out_dir = f"pick_place_dice_evalpoints"
    os.makedirs(out_dir,exist_ok=True)
    image.save(f"{out_dir}/{args.idx}.jpg")
    
    cams.close()
