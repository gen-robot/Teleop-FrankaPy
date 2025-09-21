import os
from frankapy import FRANKAPY_PATH
ASSETS_PATH = os.path.join(FRANKAPY_PATH, "assets")
PANDA_URDF_PATH = os.path.join(ASSETS_PATH, "panda", "panda_v3.urdf")

URDF_GRIPPER_OPEN = 0.04
URDF_GRIPPER_CLOSE = 0