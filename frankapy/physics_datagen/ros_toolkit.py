import rospy
import numpy as np
import open3d as o3d
from std_msgs.msg import Float64MultiArray
from frankapy import FrankaArm
from geometry_msgs.msg import Transform, Vector3, Quaternion
from scipy.spatial.transform import Rotation as R

class Ros_listener:
    def __init__(self):
        self.joint_state = None
        self.ee_pose = None
        self.ee_velocity = None
        self.joint_state_subscriber = rospy.Subscriber('/franka/joint_states', Float64MultiArray, self.state_callback_joint_state)
        self.ee_pose_subscriber = rospy.Subscriber('/franka/end_effector_pose', Transform, self.state_callback_ee_pose)
        #self.ee_velocity_subscriber = rospy.Subscriber('/franka/end_effector_velocity', Float64MultiArray, self.state_callback_ee_velocity)

    def state_callback_joint_state(self, msg):
        self.joint_state = msg.data

    def state_callback_ee_pose(self, msg):
        self.ee_pose = msg

    def state_callback_ee_velocity(self, msg):
        self.ee_velocity = msg.data

class Ros_publisher:
    def __init__(self, arm=None, vis_pose=False):
        # 初始化 Franka Panda 机器人接口
        if arm is None:
            self.arm = FrankaArm()
        else:
            self.arm = arm
        self.vis_pose = vis_pose

        # 如果需要可视化，则初始化 Open3D
        if self.vis_pose:
            self.o3d_vis = o3d.visualization.Visualizer()
            self.o3d_vis.create_window()
            self.ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.first_frame = True

        rospy.loginfo("initing...")
        # 将机器人移动到中性位（确保中性位在你的工作空间内有效）
        #self.arm.move_to_neutral()

        # 获取初始状态
        self.current_joint_state = self.arm.get_joints()  # 例如返回 7 个关节角（单位：弧度）
        self.joint_state = self.current_joint_state[:]

        # 获取末端执行器位姿，假设返回 [x, y, z, roll, pitch, yaw]
        self.current_ee_pose = self.arm.get_pose()
        self.ee_pose = self.current_ee_pose

        # 初始化 ROS 节点
        #rospy.init_node('franka_arm_controller', anonymous=True)

        # 发布当前关节状态
        self.joint_state_publisher = rospy.Publisher('/franka/joint_states', Float64MultiArray, queue_size=1)

        # 发布末端执行器位姿
        self.end_effector_publisher = rospy.Publisher('/franka/end_effector_pose', Transform, queue_size=1)
        #self.end_effector_velocity_publisher = rospy.Publisher('/franka/end_effector_velocity', Float64MultiArray, queue_size=1)

    def read_joint_state(self):
        # 更新当前关节状态与末端执行器位姿
        #rospy.loginfo("reading...")
        self.current_joint_state = self.arm.get_joints()
        self.joint_state = self.current_joint_state[:]

        self.current_ee_pose = self.arm.get_pose()
        self.ee_pose = self.current_ee_pose

        # 构造并发布关节状态消息
        joint_state_msg = Float64MultiArray()
        joint_state_msg.data = self.joint_state
        self.joint_state_publisher.publish(joint_state_msg)
        #rospy.loginfo(self.joint_state)
        # 构造并发布末端执行器位姿消息
        ee_pose_msg = Transform()

        # Extract translation (assuming self.ee_pose.translation is a numpy array)
        translation = self.ee_pose.translation
        # Convert numpy array to Vector3
        ee_pose_msg.translation = Vector3(translation[0], translation[1], translation[2])

        # Extract rotation matrix (assuming self.ee_pose.rotation is a 3x3 numpy matrix)
        rotation_matrix = self.ee_pose.rotation

        # Convert the rotation matrix to a quaternion
        rotation = R.from_matrix(rotation_matrix).as_quat()  # Returns [x, y, z, w]

        # Assign the quaternion to the Transform message
        ee_pose_msg.rotation = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
        # rospy.loginfo("Published end effector pose: %s", ee_pose_msg)
        self.end_effector_publisher.publish(ee_pose_msg)
        # 构造并发布末端执行器速度消息
        # ee_velocity = self.arm.get_ee_velocity()
        # ee_velocity_msg = Float64MultiArray()
        # ee_velocity_msg.data = ee_velocity
        # self.end_effector_velocity_publisher.publish(ee_velocity_msg)

    def _ee_pose_to_matrix(self, ee_pose):
        # 将末端执行器位姿（[x, y, z, roll, pitch, yaw]）转换为齐次变换矩阵
        pos = np.array(ee_pose[:3])
        rpy = ee_pose[3:]
        R = o3d.geometry.get_rotation_matrix_from_xyz(rpy)
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = pos
        return mat

    def _vis_pose(self, ee_pose):
        # 更新 Open3D 中显示的末端执行器坐标系
        if self.first_frame:
            self.ee_frame.transform(self._ee_pose_to_matrix(ee_pose))
            self.o3d_vis.add_geometry(self.base_frame)
            self.o3d_vis.add_geometry(self.ee_frame)
            self.first_frame = False
        else:
            self.o3d_vis.remove_geometry(self.ee_frame)
            self.ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            self.ee_frame.transform(self._ee_pose_to_matrix(ee_pose))
            self.o3d_vis.add_geometry(self.ee_frame)
            self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()

    def run(self):
        rate = rospy.Rate(50)  # 设定循环频率为 10 Hz
        while not rospy.is_shutdown():
            self.read_joint_state()
            # if self.vis_pose:
            #     self._vis_pose(self.ee_pose)
            rate.sleep()

    def shutdown(self):
        
        #self.arm.shutdown()
        rospy.loginfo("Franka arm controller shutdown.")


def run_publisher(arm):
    try:
        ros_publisher = Ros_publisher(arm, vis_pose=False)
        ros_publisher.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        ros_publisher.shutdown()


if __name__ == '__main__':
    run_publisher()
