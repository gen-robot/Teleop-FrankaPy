"""
Test script for verifying the IK solver integration with FrankaPy.
This script tests the kinematics solver without running the actual robot.
"""
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Add ManiSkill path
MANISKILL_PATH = "/home/pancake/Documents/ManiSkill"
if MANISKILL_PATH not in sys.path:
    sys.path.insert(0, MANISKILL_PATH)

from kinematics_test.standalone_kinematics import StandaloneKinematics, Pose as IKPose


def quaternion_to_wxyz(quat_xyzw):
    """Convert quaternion from (x,y,z,w) to (w,x,y,z) format."""
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def rotation_matrix_to_quaternion_wxyz(rot_matrix):
    """Convert rotation matrix to quaternion in (w,x,y,z) format."""
    r = R.from_matrix(rot_matrix)
    quat_xyzw = r.as_quat()  # scipy returns (x,y,z,w)
    return quaternion_to_wxyz(quat_xyzw)


def test_fk_ik_consistency():
    """Test that FK and IK are consistent."""
    print("=" * 80)
    print("Test 1: FK-IK Consistency")
    print("=" * 80)
    
    # Initialize IK solver
    import os
    urdf_path = os.path.join(MANISKILL_PATH, "mani_skill/assets/robots/panda/panda_v3.urdf")
    
    panda_arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3",
        "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    ik_solver = StandaloneKinematics(
        urdf_path=urdf_path,
        end_link_name="panda_hand_tcp",
        joint_names=panda_arm_joint_names,
        device=device,
        dtype=torch.float32,
    )
    
    # Test with Panda home configuration
    home_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    print(f"Test joints (home): {home_joints}")
    
    # Compute FK
    home_joints_torch = torch.tensor(home_joints, dtype=torch.float32, device=device)
    ee_pose = ik_solver.compute_fk(home_joints_torch)
    
    print(f"\nFK Result:")
    print(f"  Position: {ee_pose.p.cpu().numpy()[0]}")
    print(f"  Quaternion (wxyz): {ee_pose.q.cpu().numpy()[0]}")
    
    # Now compute IK to get back to the same joints
    ik_result = ik_solver.compute_ik(ee_pose, initial_qpos=home_joints_torch)
    
    if ik_result is not None:
        print(f"\nIK Result:")
        print(f"  Joints: {ik_result}")
        
        # Compute error
        joint_error = np.linalg.norm(ik_result - home_joints)
        print(f"\nJoint error: {joint_error:.6f} rad")
        
        # Verify with FK again
        fk_verify = ik_solver.compute_fk(torch.tensor(ik_result, dtype=torch.float32, device=device))
        pos_error = np.linalg.norm(fk_verify.p.cpu().numpy()[0] - ee_pose.p.cpu().numpy()[0])
        print(f"Position error: {pos_error:.6f} m")
        
        if joint_error < 0.01 and pos_error < 0.001:
            print("\n✓ Test PASSED: FK-IK consistency verified")
            return True
        else:
            print("\n✗ Test FAILED: Errors too large")
            return False
    else:
        print("\n✗ Test FAILED: IK returned None")
        return False


def test_delta_pose_ik():
    """Test IK with small delta poses (simulating teleoperation)."""
    print("\n" + "=" * 80)
    print("Test 2: Delta Pose IK (Teleoperation Simulation)")
    print("=" * 80)
    
    # Initialize IK solver
    import os
    urdf_path = os.path.join(MANISKILL_PATH, "mani_skill/assets/robots/panda/panda_v3.urdf")
    
    panda_arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3",
        "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ik_solver = StandaloneKinematics(
        urdf_path=urdf_path,
        end_link_name="panda_hand_tcp",
        joint_names=panda_arm_joint_names,
        device=device,
        dtype=torch.float32,
    )
    
    # Start from home configuration
    current_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    print(f"Starting joints: {current_joints}\n")
    
    # Simulate 10 steps of small pose changes
    num_steps = 10
    delta_pos = np.array([0.01, 0.0, 0.0])  # 1cm in x direction per step
    
    success_count = 0
    max_pos_error = 0.0
    
    for step in range(num_steps):
        # Get current EE pose
        current_joints_torch = torch.tensor(current_joints, dtype=torch.float32, device=device)
        current_ee_pose = ik_solver.compute_fk(current_joints_torch)
        current_pos = current_ee_pose.p.cpu().numpy()[0]
        current_quat = current_ee_pose.q.cpu().numpy()[0]
        
        # Apply delta position
        target_pos = current_pos + delta_pos
        target_quat = current_quat  # Keep orientation same
        
        # Create target pose
        target_pose = IKPose(
            p=torch.tensor([target_pos], dtype=torch.float32, device=device),
            q=torch.tensor([target_quat], dtype=torch.float32, device=device)
        )
        
        # Compute IK with warm start
        ik_result = ik_solver.compute_ik(target_pose, initial_qpos=current_joints_torch)
        
        if ik_result is not None:
            # Verify solution
            fk_verify = ik_solver.compute_fk(torch.tensor(ik_result, dtype=torch.float32, device=device))
            pos_error = np.linalg.norm(fk_verify.p.cpu().numpy()[0] - target_pos)
            max_pos_error = max(max_pos_error, pos_error)
            
            print(f"Step {step+1}: IK success, pos_error = {pos_error:.6f} m")
            success_count += 1
            
            # Update current joints for next iteration
            current_joints = ik_result
        else:
            print(f"Step {step+1}: IK failed")
    
    success_rate = 100.0 * success_count / num_steps
    print(f"\n--- Results ---")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Max position error: {max_pos_error:.6f} m")
    
    if success_rate >= 90.0 and max_pos_error < 0.001:
        print("\n✓ Test PASSED: Delta pose IK working well")
        return True
    else:
        print("\n✗ Test FAILED: Success rate or error too high")
        return False


def test_rotation_matrix_conversion():
    """Test rotation matrix to quaternion conversion."""
    print("\n" + "=" * 80)
    print("Test 3: Rotation Matrix to Quaternion Conversion")
    print("=" * 80)
    
    # Create a test rotation matrix (45 degrees around z-axis)
    angle = np.pi / 4
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    
    print(f"Test rotation matrix (45° around z):")
    print(rot_matrix)
    
    # Convert to quaternion
    quat_wxyz = rotation_matrix_to_quaternion_wxyz(rot_matrix)
    print(f"\nQuaternion (wxyz): {quat_wxyz}")
    
    # Convert back to rotation matrix
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])  # xyzw format
    rot_matrix_back = r.as_matrix()
    
    # Compute error
    matrix_error = np.linalg.norm(rot_matrix - rot_matrix_back)
    print(f"\nRound-trip error: {matrix_error:.10f}")
    
    if matrix_error < 1e-6:
        print("\n✓ Test PASSED: Rotation conversion is correct")
        return True
    else:
        print("\n✗ Test FAILED: Round-trip error too large")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("IK Solver Integration Test Suite")
    print("=" * 80 + "\n")
    
    results = []
    
    # Test 1: FK-IK consistency
    results.append(("FK-IK Consistency", test_fk_ik_consistency()))
    
    # Test 2: Delta pose IK
    results.append(("Delta Pose IK", test_delta_pose_ik()))
    
    # Test 3: Rotation conversion
    results.append(("Rotation Conversion", test_rotation_matrix_conversion()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! IK solver is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
