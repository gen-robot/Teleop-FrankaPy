import rospy
import os
import time
import json
import tyro
import threading
import numpy as np
import pandas as pd
import open3d as o3d
from termcolor import cprint
from dataclasses import dataclass
import matplotlib.pyplot as plt
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage

from utils import get_traj, plot_trajectory
from robot_constants import TEST_JOINT_RANGE, REAL_K_GAINS, REAL_D_GAINS, INIT_GOAL_POSE, REAL_FRANKA_JOINT_LIMITS

@dataclass
class Args:
    """Parameters for the data collection script for system identification."""

    show_plot: bool = False
    """Whether to display the generated trajectory plot."""

    trajectory_type: str = "sin"
    """The type of trajectory to generate (e.g., 'sin', 'step', 'valid', 'combined', 'policy'.)."""

    joint_idx: int = 0
    """The index of the joint to be identified."""

    ctrl_freq: int = 10
    """Control frequency in Hz."""

    output: str = "csv/"
    """The directory path for output data and parameters."""

class RealtimeJointSysIdController:
    def __init__(self, state_update_freq=100, ctrl_freq=10):
        self.real_arm = FrankaArm('realtime_sysid_controller')
        
        # initialize lock and event for realtime control.
        self.current_joint_state = None
        self.state_lock = threading.Lock()  # create a lock, for shared data read and write.
        self.stop_daemon_event = threading.Event() # create an event, to inform the daemon thread stop.

        self.ctrl_rate = rospy.Rate(ctrl_freq)
        self.ctrl_duration = 1.0 / ctrl_freq
        self.state_update_rate = rospy.Rate(state_update_freq)
        
        vis_pose = False
        self.vis_pose = vis_pose

        self._read_robot_limits()
        self.state_updater_thread = threading.Thread(target=self._state_update_loop, daemon=True)
        rospy.loginfo("controller initialized.")

        if self.vis_pose:
            self.o3d_vis = o3d.visualization.Visualizer()
            self.o3d_vis.create_window()
            self.base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.o3d_vis.add_geometry(self.base_frame) # fixed


    def _state_update_loop(self):
        rospy.loginfo("daemon thread started, updating state...")
        while not self.stop_daemon_event.is_set():
            try:
                joints = self.real_arm.get_joints()

                with self.state_lock:
                    self.current_joint_state = joints
                
                self.state_update_rate.sleep()
      
            except rospy.ROSInterruptException:
                rospy.loginfo("ROS interrupted, stopping state update thread")
                break
            except Exception as e:
                rospy.logerr(f"daemon thread error: {e}")
                rospy.sleep(min(2.0, 0.1 * 2**getattr(self, '_error_count', 0)))
                self._error_count = getattr(self, '_error_count', 0) + 1

        rospy.loginfo("daemon thread stopped.")

    def get_latest_joint_state(self):
        with self.state_lock:
            retry_count = 0
            while self.current_joint_state is None:
                if retry_count > 50: 
                    rospy.logerr("Timeout waiting for joint state!")
                    raise TimeoutError("Failed to get joint state after multiple retries.")
                rospy.logwarn("waiting for the joint state first reading...")
                rospy.sleep(0.1)
                retry_count += 1
            return self.current_joint_state.copy()

    def _vis_pose(self, ee_pose):
        # update ee pose frame in Open3D viewer
        # ee_pose shape (4,4)
        self.ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.ee_frame.transform(ee_pose)
        self.o3d_vis.add_geometry(self.ee_frame)
        self.o3d_vis.poll_events()
        self.o3d_vis.update_renderer()
        self.o3d_vis.remove_geometry(self.ee_frame)

    def _read_robot_limits(self):
        self.MIN_POS = np.array(TEST_JOINT_RANGE["lower"])
        self.MAX_POS = np.array(TEST_JOINT_RANGE["upper"])

        cprint(f"MIN_POS: {self.MIN_POS}", "light_cyan", "on_blue")
        cprint(f"MAX_POS: {self.MAX_POS}", "light_cyan", "on_blue")

    def start_backend_state_update(self):
        self.stop_daemon_event.clear()
        if hasattr(self, 'state_updater_thread') and self.state_updater_thread.is_alive():
            rospy.logwarn("State update thread already running")
            return
        if self.state_updater_thread._started.is_set(): # start before but stop
            rospy.loginfo("Recreating thread object")
            self.state_updater_thread = threading.Thread(target=self._state_update_loop, daemon=True)
        self.state_updater_thread.start()
        rospy.loginfo("State update thread started")


    def stop(self):
        """ stop the daemon thread safely """
        rospy.loginfo("stopping the daemon thread...")
        self.stop_daemon_event.set()
        if self.state_updater_thread.is_alive():
            self.state_updater_thread.join(timeout=3) # wait for daemon thread stop, max 3 seconds
            if self.state_updater_thread.is_alive():
                rospy.logwarn("Daemon thread did not stop in time, force stopping.")
        rospy.loginfo("daemon thread stopped.")

    def step_joint_action_real(self, id, action, init_time):
        '''
        Pass the joint action to real robot
        '''
        try:
            traj_gen_proto_msg = JointPositionSensorMessage(
                id=id, timestamp=rospy.Time.now().to_time() - init_time, 
                joints=action
            )
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
            )
            self.real_arm.publish_sensor_values(ros_msg)
            cprint(f"Published real action: {action}", "green")
            rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))        
        except Exception as e:
            rospy.logerr(f"Failed to publish action: {e}")

    def run_trajectory_experiment(self, trajectory_type, joint_idx, param, show_plot=False, output="csv/"):
        try:
            self.start_backend_state_update()

            ''' -------------------------------------- GET COMMAND FOR SI -------------------------------------- '''

            lower = self.MIN_POS[joint_idx]
            upper = self.MAX_POS[joint_idx]
            goal_pos_list = get_traj(trajectory_type, lower=lower, upper=upper)

            ''' -------------------------------------- MOVE TO INITIAL POSE -------------------------------------- '''
            # Move to initial pose
            self.real_arm.goto_joints(np.array(INIT_GOAL_POSE[:7]))
            self.real_arm.open_gripper()
            rospy.sleep(1.0)
            self.real_current_joint_state = self.real_arm.get_joints()
            input("Hand on power switch... Press Enter to continue...")

            ''' -------------------------------------- INITIAL START DYNAMIC -------------------------------------- '''

            buffer_time = 20
            init_joint_state = self.get_latest_joint_state()
            if init_joint_state is None:
                rospy.logerr("Failed to get initial joint state!")
                raise RuntimeError("Initial joint state is None, cannot proceed with trajectory execution.")
            self.real_arm.goto_joints(init_joint_state, 
                                        duration=20, 
                                        k_gains=param["kp"],
                                        d_gains=param["kv"],
                                        dynamic=True, 
                                        buffer_time=buffer_time,
                                        ignore_virtual_walls=True) 
            init_time = rospy.Time.now().to_time() # to_sec()?
            start_time = time.time()
            
            # initialize data recording list
            goal_positions = []
            actual_positions = []
            t_history = []
            actual_vels = []

            ''' -------------------------------------- CONTROL LOOP -------------------------------------- '''
            
            # realtime control loop
            for i, pos in enumerate(goal_pos_list):
                init_time = rospy.Time.now().to_time() if i==0 else init_time # @bingwen 1 or 0?

                # only update one of initial joints.
                action = init_joint_state.copy()
                goal_pos = np.clip(pos, lower, upper)
                action[joint_idx] = goal_pos

                # Not blocking
                self.step_joint_action_real(i, action, init_time)

                # get joint state from daemon thread. not blocking
                current_joints = self.get_latest_joint_state()
                if current_joints is None:
                    rospy.logerr(" ================= unable to accquire valid joint state, skip recording this time ================= ")
                    continue
                
                # data recording
                # Placeholder for actual velocity feedback, if we don't have speed feedback, using 0 intead.
                signed_current_vel = 0
                # if not i == len(goal_pos_list)-1: # @bingwen to fix.
                #     goal_positions.append(goal_pos)
                goal_positions.append(action)
                actual_positions.append(current_joints)
                actual_vels.append(signed_current_vel) # maybe can get from frankapy
                t_history.append(time.time() - start_time)

                # strictly control frequency
                self.ctrl_rate.sleep()

                # For strict frequency control:
                # while rospy.Time.now().to_time() - init_time < self.ctrl_duration * (i + 1):
                #     rospy.sleep(0.0001)

            if not (len(actual_positions)==len(actual_vels)==len(t_history)==len(goal_positions)):
                rospy.loginfo("Data lengths do not match! Check your control loop.")

            rospy.loginfo("trajectory execution completed.")
            rospy.sleep(1.0)

            # save data to csv, save all joints
            df = pd.DataFrame({
                't': np.array(t_history),
                'init_qpos': np.array(init_joint_state).tolist(),
                'goal_position': np.array(goal_positions).tolist(),
                'actual_position': np.array(actual_positions).tolist(),
                'actual_velocity': np.array(actual_vels),
            })
            filename = f"joint:{joint_idx}_trajType:{trajectory_type}"
            data_filename = f"{filename}.csv"
            os.makedirs(output, exist_ok=True)
            df.to_csv(os.path.join(output, data_filename), index=False)
            print(f"Data saved to {data_filename}")

            # plot the trajectory
            plot_trajectory(t_history, actual_positions, goal_positions, actual_vels, filename, output, show_plot)
            print(f"Time taken: {time.time() - start_time}s")

        except KeyboardInterrupt:
            rospy.loginfo("Experiment interrupted by user")
        except Exception as e:
            rospy.logerr(f"Experiment failed with error: {e}")
            raise
        finally:
            self.real_arm.stop_skill()
            self.stop()
            rospy.loginfo("system identification experiment completed.")


def main():
    args = tyro.cli(Args)

    """Main execution function that accepts an instance of the Args class."""
    cprint(f"Starting system identification trajectory generation...\n{args}", "yellow")
    cprint(args, "yellow")
    param = {
        "kp": REAL_K_GAINS, # FC.DEFAULT_K_GAINS
        "kv": REAL_D_GAINS, # FC.DEFAULT_D_GAINS
    }
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "real_joints_param.json"), "w") as f:
        json.dump(param, f, indent=4)

    controller = RealtimeJointSysIdController(ctrl_freq=args.ctrl_freq)
    controller.run_trajectory_experiment(
        trajectory_type=args.trajectory_type,
        joint_idx=args.joint_idx,
        param=param,
        show_plot=args.show_plot,
        output=args.output
    )

if __name__ == "__main__":
    main()
