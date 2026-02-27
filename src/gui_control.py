import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

from src.robotic_arm import RoboticArm
from src.visualizer import Visualizer
from src.trajectory import TrajectoryPlanner

class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("6-DOF Robotic Arm Control Center")
        self.root.geometry("1400x900")
        
        # 核心模块初始化
        self.arm = RoboticArm(num_joints=6)
        self.viz = Visualizer(self.arm)
        self.planner = TrajectoryPlanner(self.arm)
        self.viz.setup_plot()
        
        self.current_anim = None 
        
        # --- 新增：界面就绪标志位，防止初始化时乱触发回调 ---
        self.is_ready = False 
        # --- 新增：是否正在被代码自动更新（防止滑块回调死循环） ---
        self.is_animating = False 
        
        self.create_widgets()
        
        # --- 新增：界面搭建完毕，允许响应事件 ---
        self.is_ready = True 
        
        self.update_visualization()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：控制面板
        control_frame = ttk.Frame(main_frame, width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 右侧：3D 可视化
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 1. 关节控制
        joint_frame = ttk.LabelFrame(control_frame, text="Joint Control (FK)", padding="10")
        joint_frame.pack(fill=tk.X, pady=5)
        
        self.joint_sliders = []
        self.joint_labels = []
        
        for i in range(6):
            frame = ttk.Frame(joint_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=f"J{i+1}:", width=4).pack(side=tk.LEFT)
            
            slider = ttk.Scale(frame, from_=-180, to=180, orient=tk.HORIZONTAL, 
                               command=lambda val, idx=i: self.on_slider_change(idx, val))
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            label = ttk.Label(frame, text="0.0°", width=6)
            label.pack(side=tk.LEFT)
            
            # 修复：先将对象添加到列表，确保列表长度足够
            self.joint_sliders.append(slider)
            self.joint_labels.append(label)
            
            # 最后再设置初始值，因为 set() 会立刻触发回调函数
            slider.set(0)

        # 2. 轨迹动画模块 (新增)
        anim_frame = ttk.LabelFrame(control_frame, text="Trajectory Animations", padding="10")
        anim_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(anim_frame, text="▶ Draw Pentagram (画五角星)", command=self.play_pentagram).pack(fill=tk.X, pady=4)
        ttk.Button(anim_frame, text="▶ 3D Spiral (螺旋轨迹)", command=self.play_spiral).pack(fill=tk.X, pady=4)
        ttk.Button(anim_frame, text="▶ Smooth Blending (多段平滑)", command=self.play_smooth).pack(fill=tk.X, pady=4)
        ttk.Button(anim_frame, text="⏹ Stop Animation (停止动画)", command=self.stop_animation).pack(fill=tk.X, pady=4)

        # 3. 状态显示
        status_frame = ttk.LabelFrame(control_frame, text="System Status", padding="10")
        status_frame.pack(fill=tk.X, pady=10)
        self.status_label = ttk.Label(status_frame, text="Ready", font=('Consolas', 10), foreground="blue")
        self.status_label.pack(fill=tk.X)
        
        # 4. 退出按钮 (新增)
        ttk.Button(control_frame, text="✖ Exit App (退出程序)", command=self.exit_app).pack(side=tk.BOTTOM, fill=tk.X, pady=20)

        # 5. 嵌入图像
        self.canvas = FigureCanvasTkAgg(self.viz.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_slider_change(self, joint_idx, value):
        # 1. 拦截界面未就绪
        if not getattr(self, 'is_ready', False):
            return
            
        # 2. 拦截代码自动更新触发的回调！！！（保住红线的关键）
        if getattr(self, 'is_animating', False):
            return
            
        self.stop_animation() # 手动拖动滑块时停止动画
        angle_deg = float(value)
        self.joint_labels[joint_idx].config(text=f"{angle_deg:.1f}°")
        angles = [np.deg2rad(float(s.get())) for s in self.joint_sliders]
        self.arm.set_joint_angles(angles)
        self.viz.set_trajectory_path([]) # 清除轨迹线
        self.update_visualization()

    def update_visualization(self):
        self.viz.draw_arm()
        pos = self.arm.get_end_effector_position()
        self.status_label.config(text=f"EE Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        self.canvas.draw_idle()

    # ================= 动画核心逻辑 =================
    def _run_animation_loop(self, trajectory, cartesian_path):
        """通用的动画执行器 (支持红色轨迹显示与循环播放)"""
        self.stop_animation()
        
        # 强制绘制红色目标轨迹线
        self.viz.set_trajectory_path(cartesian_path)
        
        def update(frame):
            # 获取当前帧的关节角度并更新机械臂模型
            angles = trajectory[frame]['position']
            self.viz.draw_arm(angles)
            
            # 同步更新左侧的滑块UI
            if frame % 2 == 0:
                self.is_animating = True  # <--- 锁定：告诉系统是我在动滑块，别清空红线！
                for i, angle in enumerate(angles):
                    self.joint_sliders[i].set(np.rad2deg(angle))
                    self.joint_labels[i].config(text=f"{np.rad2deg(angle):.1f}°")
                self.is_animating = False # <--- 解锁
                    
            return self.viz.arm_line, self.viz.ee_scatter, self.viz.traj_line

        dt = trajectory[1]['time'] - trajectory[0]['time'] if len(trajectory) > 1 else 0.05
        
        # 启动动画：开启 repeat=True 循环，并在每轮结束时停顿 1000 毫秒
        self.current_anim = FuncAnimation(
            self.viz.fig, update, frames=len(trajectory),
            interval=dt * 1000, blit=False, 
            repeat=True, repeat_delay=1000
        )
        self.canvas.draw()

    def play_pentagram(self):
        self.status_label.config(text="Calculating Pentagram Trajectory...")
        self.root.update()
        
        center = np.array([0.4, 0.0, 0.3])
        radius = 0.15
        angles = [i * 2 * np.pi / 5 for i in range(5)]
        vertices = [center + np.array([0, radius * np.sin(a), radius * np.cos(a)]) for a in angles]
        order = [0, 2, 4, 1, 3, 0]
        pts = [vertices[i] for i in order]
        
        full_trajectory = []
        for i in range(len(pts) - 1):
            segment = self.planner.plan_cartesian_line(pts[i], pts[i+1], num_points=15)
            full_trajectory.extend(segment)
            
        cart_path = [pt['cartesian_pos'] for pt in full_trajectory if 'cartesian_pos' in pt]
        self.status_label.config(text="Playing: Pentagram")
        self._run_animation_loop(full_trajectory, cart_path)

    def play_spiral(self):
        self.status_label.config(text="Calculating Spiral Trajectory...")
        self.root.update()
        
        center = np.array([0.3, 0.0, 0.1])
        radius, height, num_points = 0.1, 0.4, 80
        trajectory = []
        cart_path = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            angle = 4 * np.pi * t
            target_pos = center + np.array([radius * np.cos(angle), radius * np.sin(angle), height * t])
            
            initial_guess = trajectory[-1]['position'] if trajectory else None
            success, q = self.arm.inverse_kinematics(target_pos, initial_guess=initial_guess)
            if success:
                trajectory.append({'time': t*5.0, 'position': q})
                cart_path.append(target_pos)
                
        self.status_label.config(text="Playing: 3D Spiral")
        self._run_animation_loop(trajectory, cart_path)

    def play_smooth(self):
        self.status_label.config(text="Calculating Smooth Blending...")
        self.root.update()
        
        via_points = [
            [0.0,  0.0,  0.0, 0.0, 0.0, 0.0],
            [1.0, -0.5,  0.8, 0.0, 0.0, 0.0],
            [-0.5, 0.8, -0.5, 0.0, 0.0, 0.0],
            [0.0,  0.0,  0.0, 0.0, 0.0, 0.0]
        ]
        times = [0.0, 2.0, 4.0, 6.0]
        trajectory = self.planner.plan_multi_point_trajectory(via_points, times, num_points_per_segment=30)
        
        # 计算对应的笛卡尔路径用于绘制红线
        cart_path = []
        for pt in trajectory:
            self.arm.set_joint_angles(pt['position'])
            cart_path.append(self.arm.get_end_effector_position())
            
        self.status_label.config(text="Playing: Smooth Blending")
        self._run_animation_loop(trajectory, cart_path)

    def stop_animation(self):
        """停止当前正在播放的动画"""
        if self.current_anim is not None:
            self.current_anim.event_source.stop()
            self.current_anim = None
            self.status_label.config(text="Animation Stopped")

    def exit_app(self):
        """安全退出程序"""
        self.stop_animation()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()