import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self, robotic_arm):
        """初始化可视化器"""
        self.arm = robotic_arm
        self.fig = None
        self.ax = None
        # 图形对象
        self.arm_line = None
        self.base_scatter = None
        self.ee_scatter = None
        self.traj_line = None  # 新增：用于绘制目标轨迹线

    def setup_plot(self):
        """设置3D绘图环境"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
        self.ax.clear()
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        max_range = 1.0
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([0, max_range*1.5])
        
        self.ax.view_init(elev=20, azim=45)
        self.ax.set_title('6-DOF Robotic Arm System')
        
        # 初始化空白的绘图对象
        self.traj_line, = self.ax.plot([], [], [], 'r--', linewidth=2, alpha=0.7, label='Target Path')
        self.arm_line, = self.ax.plot([], [], [], 'o-', color='#3498db', linewidth=4, markersize=8, label='Robot Arm')
        self.base_scatter, = self.ax.plot([0], [0], [0], 'rs', markersize=10, label='Base')
        self.ee_scatter, = self.ax.plot([], [], [], 'g^', markersize=10, label='End Effector')
        self.ax.legend()

    def set_trajectory_path(self, cartesian_points):
            """设置并绘制预定的末端笛卡尔轨迹线"""
            if cartesian_points is not None and len(cartesian_points) > 0:
                pts = np.array(cartesian_points)
                # 修复：使用更原生的 3D 赋值方法，并确保有数据
                self.traj_line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
                self.traj_line.set_alpha(0.8) # 确保透明度可见
            else:
                self.traj_line.set_data_3d([], [], [])
                
            # 强制通知画布重绘该图层
            if self.fig is not None:
                self.fig.canvas.draw_idle()
            
    def draw_arm(self, joint_angles=None):
        """绘制机械臂连杆和关节 (高性能更新)"""
        if joint_angles is not None:
            self.arm.set_joint_angles(joint_angles)
            
        _, joint_positions = self.arm.forward_kinematics()
        joint_positions = np.array(joint_positions)
        
        xs = joint_positions[:, 0]
        ys = joint_positions[:, 1]
        zs = joint_positions[:, 2]
        
        self.arm_line.set_data(xs, ys)
        self.arm_line.set_3d_properties(zs)
        
        self.ee_scatter.set_data([xs[-1]], [ys[-1]])
        self.ee_scatter.set_3d_properties([zs[-1]])

    def show(self):
        plt.show()