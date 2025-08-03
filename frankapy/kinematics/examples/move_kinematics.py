import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
from kinematics.frankapy_utils import IKSolver, DynamicJointTrajectoryPublisher, generate_pose_trajectory

import rospy


def execute_static_trajectory(fa, pose_traj, T, dt):
    """Execute trajectory using static IK solving (solve all poses at once)"""
    rospy.loginfo('Executing static trajectory')
    
    # Initialize IK solver
    ik_solver = IKSolver()
    
    # Get current joint state
    current_joints = fa.get_joints()
    
    # Solve entire trajectory at once
    joints_traj = ik_solver.solve_trajectory_list(current_joints, pose_traj)
    rospy.loginfo(f'Converted to joint trajectory with {len(joints_traj)} points')
    
    # Initialize trajectory publisher
    traj_publisher = DynamicJointTrajectoryPublisher()
    rate = rospy.Rate(1 / dt)
    
    # Start dynamic execution with first joint position
    traj_publisher.start_dynamic_execution(fa, joints_traj[1], T)
    
    # Publish remaining joint targets
    for i in range(2, len(joints_traj)):
        traj_publisher.publish_joint_target(joints_traj[i], i)
        rate.sleep()
    
    # Terminate execution
    traj_publisher.terminate_execution()


if __name__ == "__main__":
    try:
        rospy.loginfo('Initializing FrankaArm and resetting joints')
        fa = FrankaArm()
        fa.reset_joints()
        
        # Generate pose trajectory
        rospy.loginfo('Generating pose trajectory')
        pose_traj, T, dt = generate_pose_trajectory(fa)
        rospy.loginfo(f'Generated pose trajectory with {len(pose_traj)} points')
        
        # Execute static trajectory
        execute_static_trajectory(fa, pose_traj, T, dt)
        
        rospy.loginfo('Move kinematics completed successfully!')
        
    except Exception as e:
        rospy.logerr(f'Error in move_kinematics: {str(e)}')
        raise
    
    rospy.loginfo('Done')