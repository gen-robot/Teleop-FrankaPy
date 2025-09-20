import os
import yourdfpy
import numpy as np
import pyroki as pk

try:
    from kinematics.solution.traj_optimization import solve_trajopt
    from kinematics.solution.solve_ik import solve_ik
    PYROKI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import PyRoKi solutions: {e}")
    print("Falling back to basic implementations")
    PYROKI_AVAILABLE = False
    
    # Define dummy functions for fallback
    def solve_trajopt(*args, **kwargs):
        raise NotImplementedError("PyRoKi trajectory optimization not available")
    
    def solve_ik(*args, **kwargs):
        raise NotImplementedError("PyRoKi IK solver not available")

def visualize(ee_translation):
    """Visualize trajectory points in 3D."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Convert to numpy array if needed
    if isinstance(ee_translation, list):
        ee_translation = np.array(ee_translation)
    
    # Extract x, y, z coordinates
    x = ee_translation[:, 0]
    y = ee_translation[:, 1]
    z = ee_translation[:, 2]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(x, y, z, c='r', marker='o', label='Points')

    # Add axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax.set_title('3D Point Visualization')
    ax.legend()
    
    # Get min/max values for each axis
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    # Calculate maximum range
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Calculate axis centers
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    # Set axis ranges for equal scaling
    ax.set_xlim([x_center - max_range / 2, x_center + max_range / 2])
    ax.set_ylim([y_center - max_range / 2, y_center + max_range / 2])
    ax.set_zlim([z_center - max_range / 2, z_center + max_range / 2])
    plt.show()

def init_pyroki(args):
    """Initialize PyRoKi robot model and collision checker."""
    # Default URDF path - adjust based on your setup
    if hasattr(args, 'urdf') and args.urdf:
        urdf_path = args.urdf
    else:
        # Try to find URDF in common locations
        possible_paths = [
            "./assets/panda/panda_v3.urdf",
            "robot/franka_description/franka_panda.urdf",
            "/opt/ros/noetic/share/franka_description/robots/panda/panda.urdf",
        ]
        urdf_path = None
        for path in possible_paths:
            if os.path.exists(path):
                urdf_path = path
                break
        
        if urdf_path is None:
            raise FileNotFoundError(f"Could not find URDF file. Tried: {possible_paths}")
    
    print(f"Loading URDF from: {urdf_path}")
    
    try:
        urdf = yourdfpy.URDF.load(urdf_path)
        robot = pk.Robot.from_urdf(urdf)
        
        # Initialize collision checker
        robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
        world_coll = []  # No world obstacles for now
        
        print("PyRoKi robot model loaded successfully")
        return robot, robot_coll, world_coll
        
    except Exception as e:
        print(f"Failed to load robot model: {e}")
        raise

def basic_ik_solver(robot, target_position, target_wxyz, current_joints, target_link_name="panda_hand_tcp"):
    """
    Basic IK solver using numerical optimization when PyRoKi IK is not available.
    """
    from scipy.optimize import minimize
    
    def objective(joints):
        try:
            fk_result = robot.forward_kinematics(joints)
            # Assuming end-effector is the last link
            ee_pose = fk_result[-1]  # 4x4 transformation matrix
            current_pos = ee_pose[:3, 3]
            
            # Position error
            pos_error = np.linalg.norm(current_pos - target_position)
            
            # Orientation error (simplified - just use position for now)
            return pos_error
        except:
            return 1e6  # Large penalty for invalid configurations
    
    # Use current joints as initial guess
    result = minimize(objective, current_joints, method='BFGS')
    
    if result.success:
        return result.x
    else:
        print("Basic IK optimization failed")
        return current_joints  # Return current configuration as fallback

def solve_motion(args, joint_state, ee_translation_goal, ee_orientation_goal, robot, robot_coll, world_coll=None):
    """
    Solve motion planning from current joint state to target pose using PyRoKi.
    
    Args:
        args: Arguments object with debug settings
        joint_state: Current joint configuration (7-DOF for Franka)
        ee_translation_goal: Target end-effector position [x, y, z]
        ee_orientation_goal: Target end-effector orientation [x, y, z, w]
        robot: PyRoKi Robot object
        robot_coll: PyRoKi collision checker
        world_coll: World collision objects (optional)
    
    Returns:
        tuple: (joint_trajectory, ee_position_trajectory, final_joint_state) or None if failed
    """
    if world_coll is None:
        world_coll = []
    
    # Convert inputs to numpy arrays
    joint_state = np.array(joint_state)
    ee_translation_goal = np.array(ee_translation_goal)
    ee_orientation_goal = np.array(ee_orientation_goal)
    
    # Convert quaternion from [x,y,z,w] to [w,x,y,z] format for PyRoKi
    target_wxyz = np.array([ee_orientation_goal[3], ee_orientation_goal[0], 
                           ee_orientation_goal[1], ee_orientation_goal[2]])
    
    print("Target position: ", ee_translation_goal)
    print("Target quaternion (wxyz): ", target_wxyz)
    print("Current joint state: ", joint_state)
    
    # Compute current end-effector pose for forward kinematics check
    current_fk = robot.forward_kinematics(joint_state)
    ee_link_index = -1  # Assuming last link is end-effector, adjust as needed
    current_ee_pose = current_fk[ee_link_index]
    current_ee_pos = current_ee_pose[:3, 3]  # Translation part
    print("Current EE position from FK: ", current_ee_pos)
    
    try:
        # Method 1: Try trajectory optimization for smooth motion (if available)
        target_link_name = "panda_hand_tcp"  # Adjust based on your URDF
        
        if PYROKI_AVAILABLE:
            # Parameters for trajectory optimization
            timesteps = 50
            dt = 0.02
            
            try:
                trajectory = solve_trajopt(
                    robot=robot,
                    robot_coll=robot_coll,
                    world_coll=world_coll,
                    target_link_name=target_link_name,
                    start_position=current_ee_pos,
                    start_wxyz=target_wxyz,  # Keep same orientation for start
                    end_position=ee_translation_goal,
                    end_wxyz=target_wxyz,
                    timesteps=timesteps,
                    dt=dt,
                )
                
                # Compute forward kinematics for the trajectory
                ee_trajectory = []
                for joints in trajectory:
                    fk_result = robot.forward_kinematics(joints)
                    ee_pos = fk_result[ee_link_index][:3, 3]
                    ee_trajectory.append(ee_pos.tolist())
                
                print("Trajectory optimization successful")
                
                if args.debug >= 1:
                    print("Visualizing trajectory")
                    visualize(np.array(ee_trajectory))
                
                return (trajectory.tolist(), ee_trajectory, trajectory[-1].tolist())
                
            except Exception as e:
                print(f"Trajectory optimization failed: {e}, falling back to IK")
        
        # Method 2: Try PyRoKi IK solution (if available)
        if PYROKI_AVAILABLE:
            try:
                solution = solve_ik(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_position=ee_translation_goal,
                    target_wxyz=target_wxyz,
                )
                
                # Create a simple linear interpolation trajectory
                num_steps = 20
                joint_trajectory = []
                ee_trajectory = []
                
                for i in range(num_steps + 1):
                    alpha = i / num_steps
                    interp_joints = joint_state + alpha * (solution - joint_state)
                    joint_trajectory.append(interp_joints.tolist())
                    
                    # Compute FK for this configuration
                    fk_result = robot.forward_kinematics(interp_joints)
                    ee_pos = fk_result[ee_link_index][:3, 3]
                    ee_trajectory.append(ee_pos.tolist())
                
                print("PyRoKi IK solution successful")
                
                if args.debug >= 1:
                    print("Visualizing IK trajectory")
                    visualize(np.array(ee_trajectory))
                
                return (joint_trajectory, ee_trajectory, solution.tolist())
                
            except Exception as e:
                print(f"PyRoKi IK solution failed: {e}, trying basic IK")
        
        # Method 3: Basic IK fallback
        try:
            solution = basic_ik_solver(robot, ee_translation_goal, target_wxyz, joint_state)
            
            # Create a simple linear interpolation trajectory
            num_steps = 20
            joint_trajectory = []
            ee_trajectory = []
            
            for i in range(num_steps + 1):
                alpha = i / num_steps
                interp_joints = joint_state + alpha * (solution - joint_state)
                joint_trajectory.append(interp_joints.tolist())
                
                # Compute FK for this configuration
                fk_result = robot.forward_kinematics(interp_joints)
                ee_pos = fk_result[ee_link_index][:3, 3]
                ee_trajectory.append(ee_pos.tolist())
            
            print("Basic IK solution successful")
            
            if args.debug >= 1:
                print("Visualizing basic IK trajectory")
                visualize(np.array(ee_trajectory))
            
            return (joint_trajectory, ee_trajectory, solution.tolist())
            
        except Exception as e:
            print(f"Basic IK solution also failed: {e}")
            
    except Exception as e:
        print(f"Motion planning failed: {e}")
    
    print("All motion planning methods failed")
    return None

# Backward compatibility aliases
init_curobo = init_pyroki

def solve_motion_legacy(args, joint_state, ee_translation_goal, ee_orientation_goal, motion_gen, kin_model):
    """Legacy interface for backward compatibility."""
    # motion_gen and kin_model should be (robot, robot_coll, world_coll) tuple from init_pyroki
    if isinstance(motion_gen, tuple) and len(motion_gen) >= 2:
        robot, robot_coll = motion_gen[:2]
        world_coll = motion_gen[2] if len(motion_gen) > 2 else []
        return solve_motion(args, joint_state, ee_translation_goal, ee_orientation_goal, 
                          robot, robot_coll, world_coll)
    else:
        # Assume motion_gen is robot and kin_model is robot_coll
        return solve_motion(args, joint_state, ee_translation_goal, ee_orientation_goal, 
                          motion_gen, kin_model, [])
