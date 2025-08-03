"""
Solve Basic IK Problem for Franka Panda Robot.
"""

import time
import viser
import yourdfpy
import numpy as np
import pyroki as pk
import kinematics.solution as pks
from viser.extras import ViserUrdf

def main():
    """Main function for basic IK."""

    urdf_path = "/home/pancake/Documents/pyroki/assets/panda/panda_v3.urdf"
    urdf = yourdfpy.URDF.load(
        urdf_path,
    )
    target_link_name = "panda_hand_tcp" # corresponding to urdf file.

    # Create robot.
    robot = pk.Robot.from_urdf(urdf,) # default_joint_cfg=[0, 0, 0, 0, 0, 0, 0, 0] # set initial pose

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target = server.scene.add_transform_controls(
        "/ik_target", scale=0.2, position=(0.61, 0.0, 0.56), wxyz=(0, 1, 0, 0)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    while True:
        # Solve IK.
        start_time = time.time()
        solution = pks.solve_ik(
            robot=robot,
            target_link_name=target_link_name,
            target_position=np.array(ik_target.position),
            target_wxyz=np.array(ik_target.wxyz),
        )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf.update_cfg(solution) # very important to update the urdf with the solution, get last state from this line.
        urdf_vis.update_cfg(solution) # includes urdf.update cfg.

if __name__ == "__main__":
    main()



