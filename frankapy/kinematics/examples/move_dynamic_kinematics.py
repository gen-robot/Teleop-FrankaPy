import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
from kinematics.frankapy_utils import IKSolver, DynamicJointTrajectoryPublisher, generate_pose_trajectory

import rospy


def execute_dynamic_trajectory(fa, pose_traj, T, dt):
    """Execute trajectory with real-time IK solving"""
    rospy.loginfo('Executing dynamic trajectory with real-time IK solving')
    
    # Initialize IK solver and trajectory publisher
    ik_solver = IKSolver(urdf_path="./assets/panda/panda_v3.urdf",
                         target_link_name="panda_hand_tcp")
    traj_publisher = DynamicJointTrajectoryPublisher()
    
    # Get current joint state and solve first pose
    current_joints = fa.get_joints()
    first_solution = ik_solver.solve_ik(current_joints, pose_traj[0])
    
    # Start dynamic execution
    traj_publisher.start_dynamic_execution(fa, first_solution, T)
    
    rate = rospy.Rate(1 / dt)
    
    # Process remaining poses dynamically
    for i in range(1, len(pose_traj)):
        rospy.loginfo(f'Processing pose {i+1}/{len(pose_traj)}')
        
        # Get actual current joint state
        actual_joints = fa.get_joints()
        rospy.loginfo(f'Current actual joints: {actual_joints}')
        
        # Solve IK for current target pose
        try:
            solution = ik_solver.solve_ik(actual_joints, pose_traj[i])
            rospy.loginfo(f'IK solution found for pose {i+1}: {solution}')
            print(f"delta_joints {actual_joints[:7] - solution[:7]}")
            # Publish joint target
            traj_publisher.publish_joint_target(solution, i)
            
        except Exception as e:
            rospy.logerr(f'IK solving failed for pose {i+1}: {str(e)}')
            continue
        
        rate.sleep()
    
    # Terminate execution
    traj_publisher.terminate_execution()


if __name__ == "__main__":
    try:
        rospy.loginfo('Initializing FrankaArm and resetting joints')
        fa = FrankaArm()
        fa.reset_joints()
        
        input("press enter to start ik")
        
        # Generate pose trajectory
        rospy.loginfo('Generating dynamic pose trajectory')
        pose_traj, T, dt = generate_pose_trajectory(fa)
        rospy.loginfo(f'Generated pose trajectory with {len(pose_traj)} points')
        
        # Execute dynamic trajectory
        execute_dynamic_trajectory(fa, pose_traj, T, dt)
        
        rospy.loginfo('Move dynamic kinematics completed successfully!')
        
    except Exception as e:
        rospy.logerr(f'Error in move_dynamic_kinematics: {str(e)}')
        raise
    
    rospy.loginfo('Done')