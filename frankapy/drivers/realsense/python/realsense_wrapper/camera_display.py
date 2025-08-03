import numpy as np
import cv2
import threading
import time
import os
from datetime import datetime
from pynput.keyboard import Listener, Key
import sys
import argparse
from realsense_wrapper.realsense_d435 import RealsenseAPI


'''
    python camera_display.py --width 640 --height 480 --fps 30 --save-dir ./captures

    ---
    
    from camera_display import CameraDisplay
    display = CameraDisplay()
    display.start()
    custom_frames = your_image_data # shape: (N, H, W, C)
    display.update(custom_frames)
    display.stop()
'''

class CameraDisplay:
    """
    Real-time camera display system for multiple cameras.
    Supports keyboard controls for saving frames and adjusting camera parameters.
    """
    
    def __init__(self, width=640, height=480, fps=30, save_dir="./camera_captures"):
        """
        Initialize camera display system.
        
        Args:
            width (int): Camera frame width
            height (int): Camera frame height
            fps (int): Camera FPS
            save_dir (str): Directory to save captured frames
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize camera system
        try:
            self.camera_system = RealsenseAPI(height=height, width=width, fps=fps)
            self.num_cameras = self.camera_system.get_num_cameras()
            print(f"Initialized {self.num_cameras} cameras")
        except Exception as e:
            print(f"Failed to initialize cameras: {e}")
            sys.exit(1)
        
        # Display control
        self.running = False
        self.display_mode = 'rgb'  # 'rgb', 'depth', 'rgbd'
        self.grid_cols = min(2, self.num_cameras)  # Max 2 columns
        self.grid_rows = (self.num_cameras + self.grid_cols - 1) // self.grid_cols
        
        # Calculate display window size
        self.display_width = self.width * self.grid_cols
        self.display_height = self.height * self.grid_rows
        
        # Keyboard control state
        self.save_frame_flag = False
        self.quit_flag = False
        self.exposure_adjustment = 0
        self.gain_adjustment = 0
        self.current_camera = 0  # For parameter adjustment
        
        # Threading
        self.display_thread = None
        self.keyboard_listener = None
        
        # Frame storage for external access
        self.current_frames = None
        self.frame_lock = threading.Lock()
        
        print("Camera Display initialized. Press 'h' for help.")

    def create_grid_display(self, frames):
        """
        Create a grid display from multiple camera frames.
        
        Args:
            frames (np.array): Array of shape (N, height, width, channels)
            
        Returns:
            np.array: Combined grid image
        """
        if frames is None or len(frames) == 0:
            return np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Handle different display modes
        if self.display_mode == 'depth':
            # Convert depth to displayable format
            display_frames = []
            for frame in frames:
                if len(frame.shape) == 3 and frame.shape[2] > 3:
                    # Extract depth channel from RGBD
                    depth = frame[:, :, 3]
                else:
                    depth = frame if len(frame.shape) == 2 else frame[:, :, 0]
                
                # Normalize depth for display
                depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
                display_frames.append(depth_colored)
            frames = np.array(display_frames)
        
        elif self.display_mode == 'rgb':
            # Use RGB channels only
            if frames.shape[-1] > 3:
                frames = frames[:, :, :, :3]
        
        # Create grid
        grid = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        for i, frame in enumerate(frames):
            if i >= self.num_cameras:
                break
                
            row = i // self.grid_cols
            col = i % self.grid_cols
            
            y_start = row * self.height
            y_end = y_start + self.height
            x_start = col * self.width
            x_end = x_start + self.width
            
            # Ensure frame is in correct format
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] > 3:
                frame = frame[:, :, :3]
            
            # Add camera label
            frame_with_label = frame.copy()
            cv2.putText(frame_with_label, f"Cam {i+1}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Highlight current camera for parameter adjustment
            if i == self.current_camera:
                cv2.rectangle(frame_with_label, (0, 0), (self.width-1, self.height-1), 
                             (0, 255, 255), 3)
            
            grid[y_start:y_end, x_start:x_end] = frame_with_label
        
        return grid

    def update(self, frames=None):
        """
        Update display with new frames.
        
        Args:
            frames (np.array, optional): External frames to display. 
                                       If None, captures from cameras.
        """
        if frames is None:
            # Capture from cameras
            if self.display_mode == 'rgb':
                frames = self.camera_system.get_rgb()
            elif self.display_mode == 'depth':
                frames = self.camera_system.get_depth()
            elif self.display_mode == 'rgbd':
                frames = self.camera_system.get_rgbd()
        
        # Store frames for external access
        with self.frame_lock:
            self.current_frames = frames.copy() if frames is not None else None
        
        # Handle save frame request
        if self.save_frame_flag:
            self.save_current_frames(frames)
            self.save_frame_flag = False
        
        return frames

    def display_loop(self):
        """Main display loop running in separate thread."""
        cv2.namedWindow('Camera Display', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Display', self.display_width, self.display_height)
        
        while self.running:
            try:
                frames = self.update()
                
                if frames is not None:
                    grid_image = self.create_grid_display(frames)
                    
                    # Add status text
                    status_text = f"Mode: {self.display_mode} | Current Cam: {self.current_camera+1} | Press 'h' for help"
                    cv2.putText(grid_image, status_text, (10, grid_image.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Camera Display', grid_image)
                
                # Check for window close or ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or self.quit_flag:  # ESC key
                    break
                    
            except Exception as e:
                print(f"Display error: {e}")
                break
            
            time.sleep(1.0 / self.fps)
        
        cv2.destroyAllWindows()
        self.running = False

    def save_current_frames(self, frames):
        """Save current frames to disk with camera parameters and organized by timestamp."""
        if frames is None:
            print("No frames to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create timestamped directory
        save_session_dir = os.path.join(self.save_dir, f"capture_{timestamp}")
        os.makedirs(save_session_dir, exist_ok=True)
        
        # Get camera parameters for all cameras
        all_camera_params = self.camera_system.get_all_cameras_params()
        
        # Save parameters info as text file
        params_file = os.path.join(save_session_dir, "camera_params.txt")
        with open(params_file, 'w') as f:
            f.write(f"Capture Time: {timestamp}\n")
            f.write(f"Display Mode: {self.display_mode}\n")
            f.write(f"Resolution: {self.width}x{self.height}\n")
            f.write(f"FPS: {self.fps}\n\n")
            
            for cam_name, params in all_camera_params.items():
                f.write(f"{cam_name} parameters:\n")
                for param_name, param_value in params.items():
                    f.write(f"  {param_name}: {param_value}\n")
                f.write("\n")
        
        for i, frame in enumerate(frames):
            # Get camera parameters for filename
            cam_params = all_camera_params.get(f'cam{i+1}', {})
            
            # Create parameter string for filename (only key parameters)
            param_parts = []
            if 'exp' in cam_params:
                param_parts.append(f"exp{cam_params['exp']}")
            if 'gain' in cam_params:
                param_parts.append(f"g{cam_params['gain']}")
            if 'laser' in cam_params and self.display_mode in ['depth', 'rgbd']:
                param_parts.append(f"l{cam_params['laser']}")
            if 'auto_exp' in cam_params and cam_params['auto_exp']:
                param_parts.append("auto")
            
            param_string = "_".join(param_parts) if param_parts else "default"
            
            if self.display_mode == 'depth':
                # Save depth as 16-bit PNG
                if len(frame.shape) == 3 and frame.shape[2] > 3:
                    depth_data = frame[:, :, 3]
                else:
                    depth_data = frame if len(frame.shape) == 2 else frame[:, :, 0]
                
                filename = os.path.join(save_session_dir, f"depth_cam{i+1}_{param_string}.png")
                cv2.imwrite(filename, depth_data.astype(np.uint16))
            
            elif self.display_mode == 'rgb':
                # Save RGB as standard image
                rgb_frame = frame[:, :, :3] if frame.shape[2] > 3 else frame
                filename = os.path.join(save_session_dir, f"rgb_cam{i+1}_{param_string}.jpg")
                cv2.imwrite(filename, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            
            elif self.display_mode == 'rgbd':
                # Save both RGB and depth
                rgb_frame = frame[:, :, :3]
                depth_frame = frame[:, :, 3]
                
                rgb_filename = os.path.join(save_session_dir, f"rgb_cam{i+1}_{param_string}.jpg")
                depth_filename = os.path.join(save_session_dir, f"depth_cam{i+1}_{param_string}.png")
                
                cv2.imwrite(rgb_filename, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(depth_filename, depth_frame.astype(np.uint16))
        
        print(f"Saved frames from {len(frames)} cameras to {save_session_dir}")
        print(f"Parameters saved to {params_file}")

    def on_key_press(self, key):
        """Handle key press events."""
        try:
            if hasattr(key, 'char') and key.char:
                if key.char == 's':
                    self.save_frame_flag = True
                    print("Saving current frames...")
                
                elif key.char == 'q':
                    self.quit_flag = True
                    print("Quitting...")
                
                elif key.char == 'm':
                    # Cycle through display modes
                    modes = ['rgb', 'depth', 'rgbd']
                    current_idx = modes.index(self.display_mode)
                    self.display_mode = modes[(current_idx + 1) % len(modes)]
                    print(f"Switched to {self.display_mode} mode")
                
                elif key.char == 'c':
                    # Cycle through cameras for parameter adjustment
                    self.current_camera = (self.current_camera + 1) % self.num_cameras
                    print(f"Selected camera {self.current_camera + 1} for parameter adjustment")
                
                elif key.char == 'h':
                    self.print_help()
                
                elif key.char == '=':
                    # Increase exposure
                    try:
                        options = self.camera_system.get_camera_options(self.current_camera)
                        if 'color' in options and 'exposure' in options['color']:
                            current = options['color']['exposure']['current']
                            max_val = options['color']['exposure']['max']
                            new_val = min(current + 1000, max_val)
                            self.camera_system.set_exposure(self.current_camera, new_val)
                    except Exception as e:
                        print(f"Failed to adjust exposure: {e}")
                
                elif key.char == '-':
                    # Decrease exposure
                    try:
                        options = self.camera_system.get_camera_options(self.current_camera)
                        if 'color' in options and 'exposure' in options['color']:
                            current = options['color']['exposure']['current']
                            min_val = options['color']['exposure']['min']
                            new_val = max(current - 1000, min_val)
                            self.camera_system.set_exposure(self.current_camera, new_val)
                    except Exception as e:
                        print(f"Failed to adjust exposure: {e}")
                
                elif key.char == 'a':
                    # Auto exposure
                    self.camera_system.set_exposure(self.current_camera, None)
                
                elif key.char == 'w':
                    # Auto white balance
                    self.camera_system.set_white_balance(self.current_camera, None)
        
        except AttributeError:
            # Handle special keys
            if key == Key.esc:
                self.quit_flag = True

    def on_key_release(self, key):
        """Handle key release events."""
        pass

    def print_help(self):
        """Print help information."""
        help_text = """
        Camera Display Controls:
        
        's' - Save current frames
        'q' - Quit application
        'm' - Cycle display modes (RGB -> Depth -> RGBD)
        'c' - Cycle through cameras for parameter adjustment
        '=' - Increase exposure for current camera
        '-' - Decrease exposure for current camera
        'a' - Enable auto exposure for current camera
        'w' - Enable auto white balance for current camera
        'h' - Show this help
        ESC - Quit application
        
        Current settings:
        - Mode: {mode}
        - Current camera: {cam}
        - Save directory: {save_dir}
        """.format(
            mode=self.display_mode,
            cam=self.current_camera + 1,
            save_dir=self.save_dir
        )
        print(help_text)

    def start(self):
        """Start the camera display system."""
        if self.running:
            print("Display already running")
            return
        
        self.running = True
        
        # Start keyboard listener
        self.keyboard_listener = Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        self.keyboard_listener.start()
        
        # Start display thread
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()
        
        print("Camera display started. Press 'h' for help.")

    def stop(self):
        """Stop the camera display system."""
        self.running = False
        self.quit_flag = True
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join()
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
        self.camera_system.close()
        print("Camera display stopped.")

    def get_current_frames(self):
        """Get the most recent frames (thread-safe)."""
        with self.frame_lock:
            return self.current_frames.copy() if self.current_frames is not None else None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def main():
    """Main function for standalone usage."""
    parser = argparse.ArgumentParser(description='Real-time camera display for RealSense cameras')
    parser.add_argument('--width', type=int, default=640, help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Camera height (default: 480)')
    parser.add_argument('--fps', type=int, default=30, help='Camera FPS (default: 30)')
    parser.add_argument('--save-dir', type=str, default='./camera_captures', 
                       help='Directory to save captures (default: ./camera_captures)')
    
    args = parser.parse_args()
    
    try:
        # Create and start display system
        with CameraDisplay(
            width=args.width,
            height=args.height,
            fps=args.fps,
            save_dir=args.save_dir
        ) as display:
            
            print("Camera display running. Press 'q' to quit.")
            
            # Keep main thread alive
            while display.running:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()