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


def execute_pose_trajectory_with_dynamic_ik(fa, pose_traj, robot, target_link_name, urdf, dt):
    """
    Execute pose trajectory with dynamic IK solving - solve one pose, publish it, 
    get actual joints, update URDF, then solve next pose
    """
    rospy.loginfo('Executing pose trajectory with dynamic IK solving')
    
    # Initialize URDF with current joint state
    current_joints = fa.get_joints()
    urdf.update_cfg(current_joints)
    rospy.loginfo(f'Initialized URDF with current joints: {current_joints}')
    
    # Setup publisher for dynamic trajectory
    pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
    rate = rospy.Rate(1 / dt)
    
    # Solve IK for first pose and start dynamic execution
    first_solution = pks.solve_ik(
        robot=robot,
        target_link_name=target_link_name,
        target_position=pose_traj[0].translation,
        target_wxyz=np.array([pose_traj[0].quaternion[0], pose_traj[0].quaternion[1], 
                            pose_traj[0].quaternion[2], pose_traj[0].quaternion[3]])
    )
    
    # Start dynamic execution with first joint solution
    fa.goto_joints(first_solution, duration=len(pose_traj) * dt, dynamic=True, buffer_time=10)
    init_time = rospy.Time.now().to_time()
    
    # Process remaining poses
    for i in range(1, len(pose_traj)):
        rospy.loginfo(f'Processing pose {i+1}/{len(pose_traj)}')
        
        # Get current actual joint state (simulate getting real joints during execution)
        # In real implementation, this should get actual robot state
        actual_joints = fa.get_joints()
        urdf.update_cfg(actual_joints)
        rospy.loginfo(f'Updated URDF with actual joints: {actual_joints}')
        
        # Solve IK for current target pose
        try:
            solution = pks.solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_position=pose_traj[i].translation,
                target_wxyz=np.array([pose_traj[i].quaternion[0], pose_traj[i].quaternion[1], 
                                    pose_traj[i].quaternion[2], pose_traj[i].quaternion[3]])
            )
            rospy.loginfo(f'IK solution found for pose {i+1}: {solution}')
        except Exception as e:
            rospy.logerr(f'IK solving failed for pose {i+1}: {str(e)}')
            # Use previous solution as fallback
            continue
        
        # Publish joint solution as part of dynamic trajectory
        traj_gen_proto_msg = JointPositionSensorMessage(
            id=i,
            timestamp=rospy.Time.now().to_time() - init_time,
            joints=solution
        )
        
        ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.JOINT_POSITION)
        )
        
        rospy.loginfo(f'Publishing joint trajectory: ID {traj_gen_proto_msg.id}')
        pub.publish(ros_msg)
        rate.sleep()
    
    # Send termination message
    term_proto_msg = ShouldTerminateSensorMessage(
        timestamp=rospy.Time.now().to_time() - init_time,
        should_terminate=True
    )
    ros_msg = make_sensor_group_msg(
        termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE)
    )
    pub.publish(ros_msg)
    
    rospy.loginfo('Dynamic pose trajectory execution completed')


if __name__ == "__main__":
    try:
        rospy.loginfo('Initializing FrankaArm and resetting joints')
        fa = FrankaArm()
        fa.reset_joints()
        
        rospy.loginfo('Setting up IK solver')
        robot, target_link_name, urdf = setup_ik_solver()
        
        # Generate pose trajectory
        pose_traj, T, dt = generate_pose_trajectory(fa)
        rospy.loginfo(f'Generated pose trajectory with {len(pose_traj)} points')
        
        # Execute trajectory with dynamic IK solving
        execute_pose_trajectory_with_dynamic_ik(fa, pose_traj, robot, target_link_name, urdf, dt)
        
        rospy.loginfo('Move dynamic kinematics completed successfully!')
        
    except Exception as e:
        rospy.logerr(f'Error in move_dynamic_kinematics: {str(e)}')
        raise
    
    rospy.loginfo('Done')