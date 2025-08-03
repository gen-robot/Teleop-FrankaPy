import numpy as np
import yourdfpy
import pyroki as pk
import kinematics.solution as pks

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.utils import min_jerk, min_jerk_weight
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

import rospy

class IKSolver:
    """Inverse Kinematics Solver with dynamic URDF updating"""
    
    def __init__(self, urdf_path="/home/weibingwen/Documents/assets/panda/panda_v3.urdf", 
                 target_link_name="panda_hand_tcp"):
        self.urdf_path = urdf_path
        self.target_link_name = target_link_name
        self.urdf = yourdfpy.URDF.load(urdf_path)
        self.robot = pk.Robot.from_urdf(self.urdf)
        rospy.loginfo(f'IK Solver initialized with URDF: {urdf_path}')
    
    def update_robot_state(self, joint_positions):
        """Update URDF configuration with current joint positions"""
        self.urdf.update_cfg(joint_positions)
        rospy.logdebug(f'Updated robot state with joints: {joint_positions}')
    
    def solve_ik(self, joints_state, target_pose):
        """Solve IK for a target pose"""
        try:
            self.update_robot_state(joints_state)
            solution = pks.solve_ik(
                robot=self.robot,
                target_link_name=self.target_link_name,
                target_position=target_pose.translation,
                target_wxyz=np.array([target_pose.quaternion[0], target_pose.quaternion[1], 
                                    target_pose.quaternion[2], target_pose.quaternion[3]])
            )
            return solution
        except Exception as e:
            rospy.logerr(f'IK solving failed: {str(e)}')
            raise
    
    def solve_trajectory_list(self, init_joints_state, pose_trajectory):
        """Solve IK for entire pose trajectory at once (static approach)"""
        rospy.loginfo('Converting pose trajectory to joint trajectory using batch IK')
        joints_traj = []
        joints_traj.append(init_joints_state)

        for i, pose in enumerate(pose_trajectory):
            solution = self.solve_ik(joints_traj[-1], pose)
            joints_traj.append(solution)
            
            if i % 50 == 0:
                rospy.loginfo(f'IK solving progress: {i+1}/{len(pose_trajectory)}')
        
        return joints_traj


class DynamicJointTrajectoryPublisher:
    """Publisher for dynamic joint trajectories"""
    
    def __init__(self):
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        self.init_time = None
        rospy.loginfo('Dynamic Trajectory Publisher initialized')
    
    def start_dynamic_execution(self, fa, first_joints, total_duration, buffer_time=10):
        """Start dynamic trajectory execution with first joint position"""
        fa.goto_joints(first_joints, duration=total_duration, dynamic=True, buffer_time=buffer_time)
        self.init_time = rospy.Time.now().to_time()
        rospy.loginfo('Dynamic execution started')
    
    def publish_joint_target(self, joint_positions, trajectory_id):
        """Publish a joint position target"""
        traj_gen_proto_msg = JointPositionSensorMessage(
            id=trajectory_id,
            timestamp=rospy.Time.now().to_time() - self.init_time,
            joints=joint_positions
        )
        
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
        )
        
        rospy.logdebug(f'Publishing joint trajectory: ID {trajectory_id}')
        self.pub.publish(ros_msg)
    
    def terminate_execution(self):
        """Send termination message to end dynamic execution"""
        term_proto_msg = ShouldTerminateSensorMessage(
            timestamp=rospy.Time.now().to_time() - self.init_time,
            should_terminate=True
        )
        ros_msg = make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
        )
        self.pub.publish(ros_msg)
        rospy.loginfo('Dynamic execution terminated')


def generate_pose_trajectory(fa: FrankaArm):
    
    p0 = fa.get_pose()
    p1 = p0.copy()
    
    # move up for 20 cm and rotate 30 degrees around z-axis
    T_delta = RigidTransform(
        translation=np.array([0, 0, 0.2]),
        rotation=RigidTransform.z_axis_rotation(np.deg2rad(30)),
        from_frame=p1.from_frame, to_frame=p1.from_frame
    )
    p1 = p1 * T_delta
    
    # set trajectory duration and time step
    T = 5 # s
    dt = 0.02
    ts = np.arange(0, T, dt)

    # generate min jerk trajectory weights
    weights = [min_jerk_weight(t, T) for t in ts]
    pose_traj = [p1.interpolate_with(p0, w) for w in weights]
    
    return pose_traj, T, dt
