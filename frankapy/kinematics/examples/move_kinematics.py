import numpy as np
import yourdfpy
import pyroki as pk
import kinematics.solution as pks

from autolab_core import RigidTransform
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
from franka_interface_msgs.msg import SensorDataGroup

from frankapy.utils import min_jerk, min_jerk_weight

import rospy


def setup_ik_solver():
    urdf_path = "/home/weibingwen/Documents/assets/panda/panda_v3.urdf"
    urdf = yourdfpy.URDF.load(urdf_path)
    target_link_name = "panda_hand_tcp"
    robot = pk.Robot.from_urdf(urdf)
    return robot, target_link_name, urdf


def generate_pose_trajectory(fa):

    rospy.loginfo('Generating dynamic Pose Trajectory')
    
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


def solve_pose_trajectory_to_joints(pose_traj, robot, target_link_name, urdf):
    
    rospy.loginfo('Converting pose trajectory to joint trajectory using IK')
    
    joints_traj = []
    
    for i, pose in enumerate(pose_traj):
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=pose.translation,
            target_wxyz=np.array([pose.quaternion[0], pose.quaternion[1], 
                                pose.quaternion[2], pose.quaternion[3]])  # w,x,y,z格式
        )
        joints_traj.append(solution)
        urdf.update_cfg(solution)
        
        if i % 50 == 0:
            rospy.loginfo(f'IK solving progress: {i+1}/{len(pose_traj)}')
    
    return joints_traj


def execute_joint_trajectory(fa, joints_traj, T, dt):

    rospy.loginfo('Publishing dynamic joints trajectory...')
    

    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)
    

    fa.goto_joints(joints_traj[1], duration=T, dynamic=True, buffer_time=10)
    init_time = rospy.Time.now().to_time()
    

    for i in range(2, len(joints_traj)):
        traj_gen_proto_msg = JointPositionSensorMessage(
            id=i, 
            timestamp=rospy.Time.now().to_time() - init_time,
            joints=joints_traj[i]
        )
        
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
        )
        
        rospy.loginfo('Publishing joint trajectory: ID {}'.format(traj_gen_proto_msg.id))
        pub.publish(ros_msg)
        rate.sleep()
    

    term_proto_msg = ShouldTerminateSensorMessage(
        timestamp=rospy.Time.now().to_time() - init_time, 
        should_terminate=True
    )
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
    )
    pub.publish(ros_msg)


if __name__ == "__main__":
    try:

        rospy.loginfo('Initializing FrankaArm and resetting joints')
        fa = FrankaArm()
        fa.reset_joints()
        
        rospy.loginfo('Setting up IK solver')
        robot, target_link_name, urdf = setup_ik_solver()
        

        pose_traj, T, dt = generate_pose_trajectory(fa)
        rospy.loginfo(f'Generated pose trajectory with {len(pose_traj)} points')
        

        joints_traj = solve_pose_trajectory_to_joints(pose_traj, robot, target_link_name, urdf)
        rospy.loginfo(f'Converted to joint trajectory with {len(joints_traj)} points')
        

        execute_joint_trajectory(fa, joints_traj, T, dt)
        
        rospy.loginfo('Move kinematics completed successfully!')
        
    except Exception as e:
        rospy.logerr(f'Error in move_kinematics: {str(e)}')
        raise
    
    rospy.loginfo('Done')