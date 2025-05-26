# franka-control Melodic-Noetic

### Initial preparation

``` bash 
sudo apt install ros-melodic-libfranka ros-melodic-franka-ros ros-melodic-control-msgs # you can change to noetic
conda create -n franka-teleop python=3.8 -y # 3.8 is ok, and maybe 3.10 is also ok, but we recommend 3.8 for stable
conda activate franka-teleop
pip install colcon-common-extensions "empy==3.3.4" lark-parser lxml netifaces pyyaml rosdistro vcstool setuptools
cd frankapy 
pip install -e . # you should run this before installing franka-interface
cd ..
```
## Install franka-interface in control computer [local NUC]
```bash 
cd franka-interface 
```
`You should follow the instructions` in [README.md](franka-interface/README.md)

## Install frankapy in command computer [with GPU]
```bash 
cd frankapy
```
`You should follow the instructions` in [README.md](frankapy/README.md)

`You can also follow the instructions` in [Website](https://iamlab-cmu.github.io/frankapy/install.html)

## Running the Franka Robot

1. Make sure that the Franka Robot has been unlocked in the Franka Desk GUI and has blue lights.

2. Open up a new terminal and go to the frankapy directory.
    ```bash
    cd frankapy
    sudo ufw disable # especially on C19 NUC 
    ```
    You should modify `control_pc_franka_interface_path` in `./bash_scripts/start_control_pc.sh`, to find the franka-interface.
    1. If you are running franka-interface and frankapy on the same computer, run the following command:
    ```bash 
    bash ./bash_scripts/start_control_pc.sh -i localhost
    ```
    2.  Otherwise run the following command:
    ```bash 
    bash ./bash_scripts/start_control_pc.sh -u [control-pc-username] -i [control-pc-name]
    ```
   Please see the `start_control_pc.sh` bash script for additional arguments, including specifying a custom directory for where `franka-interface` is installed on the Control PC as well as the username of the account on the Control PC. By default the username is `iam-lab`.
   
3. Open up a new terminal, enter into the same virtual environment and go to the frankapy directory. Do:
   ```bash
   # in ROS 1
   source catkin_ws/devel/setup.bash
   # in ROS 2
   source ros2_ws/install/setup.bash
   ```
   Be in the same virtualenv or Conda env that FrankaPy was installed in. Place your hand on top of the e-stop. Reset the robot pose with the following command.
   
   ```bash
   python scripts/reset_arm.py
   ```
4. See example scripts in the examples/ and scripts/ folders to learn how to use the FrankaPy python package.
   If you want to use `space_mouse to teleop and collect data`, you should follow the [instruction](./frankapy/examples/data_collection/README.md)

5. for data collection, you can read [doc](frankapy/examples/data_collection/README.md)

6. Please note that if you are using a custom gripper or no gripper, please set the with_gripper=True flag in frankapy/franka_arm.py to False as well as set the with_gripper=1 flag in bash_scripts/start_control_pc.sh to 0.

[frankapy API Documentation](https://iamlab-cmu.github.io/frankapy/)

[franka-interface API Documentation](https://iamlab-cmu.github.io/franka-interface/)