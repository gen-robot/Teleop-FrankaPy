unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PYTHON_VERSION
unset ROS_ROOT
unset ROS_PACKAGE_PATH
unset AMENT_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate conda_noetic
source $CONDA_PREFIX/setup.bash
echo "ROS Noetic Has been activated in conda environment: $CONDA_PREFIX"
# source start_conda_noetic.sh  # using this line to work
