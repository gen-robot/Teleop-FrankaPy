import os
import json
import cv2
import time
import numpy as np
from datetime import datetime
import threading
import signal
import sys
from collections import deque


class SimpleLogger:
    """
    Simple logger for recording core data: observation + action + robot state ...
    """
    
    def __init__(self, record_dir, enable_logging=True, recording_video_fps=10):
        self.record_dir = record_dir
        self.enable_logging = enable_logging
        self.video_fps = recording_video_fps
        if not self.enable_logging:
            return
            
        os.makedirs(record_dir, exist_ok=True)
        
        # Core data storage
        self.data_log = []
        
        # Video recording
        self.video_writers = {}
        self.video_queue = deque(maxlen=100)  # Prevent memory explosion
        self.recording_active = False
        self.video_thread = None
        self.init_time = time.time()
        
        print(f"[SimpleLogger] Initialized {'ON' if enable_logging else 'OFF'}")

    def log_step_data(self, step, observation, action_executed=None, command_xyz=None, command_rotation=None):
        """Record all core data for one step"""
        if not self.enable_logging:
            return
            
        # Initialize video recording if needed
        if 'images' in observation and not self.recording_active:
            self._init_video_recording(observation['images'])
        
        # Add images to video queue
        if 'images' in observation:
            self.video_queue.append(observation['images'].copy())
        
        # Create observation dict excluding image-related fields
        observation_data = {}
        for key, value in observation.items():
            # Skip any key containing 'image', 'img', 'cam', 'rgb', 'depth', 'video'
            key_lower = key.lower()
            if any(img_keyword in key_lower for img_keyword in ['image', 'img', 'cam', 'rgb', 'depth', 'video']):
                continue
            
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(value, np.ndarray):
                observation_data[key] = value.tolist()
            else:
                observation_data[key] = value
        
        # Record core data (excluding images)
        step_data = {
            'step': step,
            'timestamp': time.time() - self.init_time,
            'observation': observation_data
        }
        
        if action_executed is not None:
            step_data['action_executed'] = action_executed.tolist()
        
        if command_xyz is not None:
            step_data['command_xyz'] = command_xyz.tolist()
            
        if command_rotation is not None:
            step_data['command_rotation'] = command_rotation.tolist()
        
        self.data_log.append(step_data)

    def _init_video_recording(self, images):
        """Initialize video recording"""
        if self.recording_active:
            return
            
        num_cameras = images.shape[0]
        height, width = images.shape[1], images.shape[2]
        
        # Create video writer for each camera
        for cam_idx in range(num_cameras):
            video_path = os.path.join(self.record_dir, f'camera_{cam_idx+1}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, self.video_fps, (width, height))
            if writer.isOpened():
                self.video_writers[cam_idx] = writer
            else:
                print(f"[SimpleLogger] Failed to create video writer for camera {cam_idx+1}")
        
        # Start video recording thread
        if self.video_writers:
            self.recording_active = True
            self.video_thread = threading.Thread(target=self._video_worker, daemon=True)
            self.video_thread.start()
            print(f"[SimpleLogger] Video recording started for {len(self.video_writers)} cameras")

    def _video_worker(self):
        """Video recording worker thread"""
        while self.recording_active:
            if len(self.video_queue) > 0:
                try:
                    images = self.video_queue.popleft()
                    for cam_idx, writer in self.video_writers.items():
                        if cam_idx < len(images):
                            # Convert RGB to BGR
                            frame = cv2.cvtColor(images[cam_idx], cv2.COLOR_RGB2BGR)
                            writer.write(frame)
                except:
                    pass  # Ignore video recording errors
            else:
                time.sleep(0.01)

    def save_and_cleanup(self):
        """Save data and cleanup resources"""
        if not self.enable_logging:
            return
            
        print("[SimpleLogger] Saving data...")
        
        # Stop video recording
        self.recording_active = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2)
        
        # Close video files
        for writer in self.video_writers.values():
            writer.release()
        
        # Save core data
        if self.data_log:
            data_file = os.path.join(self.record_dir, 'session_data.json')
            with open(data_file, 'w') as f:
                json.dump(self.data_log, f, indent=2)
            
            # Save summary
            summary = {
                'total_steps': len(self.data_log),
                'cameras_recorded': len(self.video_writers),
                'session_duration': self.data_log[-1]['timestamp'] - self.data_log[0]['timestamp'] if self.data_log else 0,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(os.path.join(self.record_dir, 'summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"[SimpleLogger] Saved {len(self.data_log)} steps of data")


class VLAClient:
    """VLA client focused on connection stability"""
    
    def __init__(self, server_ip, server_port, timeout=10, max_retries=3):
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.max_retries = max_retries
        self.act_url = f"http://{server_ip}:{server_port}/act"
        self.last_success_time = None
        
    def send_request(self, observation_data):
        """Send request, returns (success, response_data, inference_time)"""
        import requests
        from requests.exceptions import RequestException, Timeout, ConnectionError
        
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.act_url, json=observation_data, timeout=self.timeout)
                
                if response.status_code == 200:
                    self.last_success_time = time.time()
                    return True, response.json(), time.time() - start_time
                else:
                    print(f"[VLAClient] Server error: {response.status_code}")
                    break
                    
            except (ConnectionError, Timeout) as e:
                print(f"[VLAClient] Attempt {attempt + 1} failed: {type(e).__name__}")
                if attempt < self.max_retries - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"[VLAClient] Unexpected error: {e}")
                break
                
        return False, None, time.time() - start_time


class GracefulExit:
    """Graceful exit handler"""
    
    def __init__(self):
        self.exit_requested = False
        self.cleanup_callbacks = []
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
    def _handle_signal(self, signum, frame):
        if not self.exit_requested:
            print(f"\n[Exit] Received signal {signum}, shutting down gracefully...")
            self.exit_requested = True
            
            # Execute cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"[Exit] Cleanup error: {e}")
    
    def add_cleanup(self, callback):
        """Add cleanup callback"""
        self.cleanup_callbacks.append(callback)
    
    def should_exit(self):
        """Check if should exit"""
        return self.exit_requested


def safe_execute(func):
    """Safe execution decorator"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[SafeExecute] Error: {e}")
            return False
    return wrapper