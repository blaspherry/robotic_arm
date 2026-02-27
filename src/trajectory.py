import numpy as np
from scipy.interpolate import CubicSpline

class TrajectoryPlanner:
    def __init__(self, robotic_arm):
        """初始化轨迹规划器"""
        self.arm = robotic_arm

    def plan_joint_trajectory(self, start_angles, end_angles, duration=2.0, method='quintic', num_points=50):
        """关节空间轨迹规划接口"""
        start_angles = np.array(start_angles)
        end_angles = np.array(end_angles)
        
        if method == 'linear':
            return self._linear_interpolation(start_angles, end_angles, duration, num_points)
        elif method == 'cubic':
            return self._cubic_interpolation(start_angles, end_angles, duration, num_points)
        elif method == 'quintic':
            return self._quintic_interpolation(start_angles, end_angles, duration, num_points)
        else:
            raise ValueError(f"未知插值方法: {method}")

    def _linear_interpolation(self, start, end, duration, num_points):
        """线性插值 (恒定速度，加速度在起点终点突变)"""
        trajectory = []
        times = np.linspace(0, duration, num_points)
        
        for t in times:
            s = t / duration # 归一化时间
            position = start + (end - start) * s
            velocity = (end - start) / duration # 速度(常数)
            acceleration = np.zeros_like(start)   # 加速度(零)
            
            trajectory.append({
                'time': t,
                'position': position,
                'velocity': velocity,
                'acceleration': acceleration
            })
        return trajectory

    def _cubic_interpolation(self, start, end, duration, num_points):
        """三次样条插值 (保证速度连续，边界速度为零)"""
        times = np.linspace(0, duration, num_points)
        trajectory = []
        key_times = [0, duration]
        
        # 为每个关节计算样条
        cs_list = []
        for i in range(len(start)):
            key_positions = [start[i], end[i]]
            # bc_type='clamped' 确保两端一阶导数(速度)为0
            cs = CubicSpline(key_times, key_positions, bc_type='clamped')
            cs_list.append(cs)
            
        for t in times:
            position = np.array([cs(t) for cs in cs_list])
            velocity = np.array([cs(t, 1) for cs in cs_list])     # 一阶导数
            acceleration = np.array([cs(t, 2) for cs in cs_list]) # 二阶导数
            
            trajectory.append({
                'time': t,
                'position': position,
                'velocity': velocity,
                'acceleration': acceleration
            })
        return trajectory

    def _quintic_interpolation(self, start, end, duration, num_points):
        """五次多项式插值 (保证位置、速度、加速度连续，边界速度和加速度为零)"""
        times = np.linspace(0, duration, num_points)
        trajectory = []
        T = duration
        
        # 对每个关节求解多项式系数 [a0, a1, a2, a3, a4, a5]
        coeffs = np.zeros((len(start), 6))
        
        for i in range(len(start)):
            q0 = start[i]
            qf = end[i]
            
            # 边界条件矩阵 (起点终点的位置、速度、加速度全为已知)
            A = np.array([
                [1, 0, 0,   0,     0,      0],
                [0, 1, 0,   0,     0,      0],
                [0, 0, 2,   0,     0,      0],
                [1, T, T**2, T**3,  T**4,   T**5],
                [0, 1, 2*T,  3*T**2, 4*T**3, 5*T**4],
                [0, 0, 2,   6*T,   12*T**2, 20*T**3]
            ])
            # 目标向量 [q(0), v(0), a(0), q(T), v(T), a(T)]
            b = np.array([q0, 0, 0, qf, 0, 0])
            coeffs[i] = np.linalg.solve(A, b)
            
        # 生成轨迹点
        for t in times:
            position = np.zeros(len(start))
            velocity = np.zeros(len(start))
            acceleration = np.zeros(len(start))
            
            for i in range(len(start)):
                a = coeffs[i]
                position[i] = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
                velocity[i] = a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4
                acceleration[i] = 2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3
                
            trajectory.append({
                'time': t,
                'position': position,
                'velocity': velocity,
                'acceleration': acceleration
            })
        return trajectory

    def plan_cartesian_line(self, start_pos, end_pos, num_points=50):
        """笛卡尔空间直线轨迹"""
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        trajectory = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            target_pos = start_pos + (end_pos - start_pos) * t
            
            # 使用上一个位置作为 IK 的初始猜测，以保证轨迹的连续性
            initial_guess = trajectory[-1]['position'] if trajectory else None
            success, angles = self.arm.inverse_kinematics(target_pos, initial_guess=initial_guess)
            
            if not success:
                print(f"警告: 笛卡尔直线路径第 {i} 个点 IK 求解失败")
                continue
                
            trajectory.append({
                'time': t,
                'position': angles,
                'cartesian_pos': target_pos
            })
        return trajectory

    def plan_cartesian_circle(self, center, radius, num_points=60, plane='xy'):
        """笛卡尔空间圆形轨迹"""
        center = np.array(center)
        trajectory = []
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            
            if plane == 'xy':
                target_pos = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            elif plane == 'xz':
                target_pos = center + np.array([radius * np.cos(angle), 0, radius * np.sin(angle)])
            elif plane == 'yz':
                target_pos = center + np.array([0, radius * np.cos(angle), radius * np.sin(angle)])
            else:
                raise ValueError("平面必须是 'xy', 'xz', 或 'yz'")
                
            initial_guess = trajectory[-1]['position'] if trajectory else None
            success, angles = self.arm.inverse_kinematics(target_pos, initial_guess=initial_guess)
            
            if success:
                trajectory.append({
                    'time': i / num_points,
                    'position': angles,
                    'cartesian_pos': target_pos
                })
        return trajectory
    
    def plan_multi_point_trajectory(self, via_points, times, num_points_per_segment=50):
        """
        多段轨迹平滑连接 (基于全局三次样条插值)
        使得机械臂在经过中间点时速度不为0，实现平滑过渡。
        """
        via_points = np.array(via_points) # 形状: (N, num_joints)
        times = np.array(times)           # 形状: (N,)
        
        # 使用 SciPy 的 CubicSpline 对所有途经点进行全局拟合
        # 默认的边界条件会保证一阶导(速度)和二阶导(加速度)在整条曲线上连续
        cs_list = []
        for i in range(self.arm.num_joints):
            # bc_type='clamped' 保证起点和终点速度为0，但中间点自由平滑
            cs = CubicSpline(times, via_points[:, i], bc_type='clamped')
            cs_list.append(cs)

        total_points = num_points_per_segment * (len(times) - 1)
        eval_times = np.linspace(times[0], times[-1], total_points)
        
        trajectory = []
        for t in eval_times:
            position = np.array([cs(t) for cs in cs_list])
            velocity = np.array([cs(t, 1) for cs in cs_list])
            acceleration = np.array([cs(t, 2) for cs in cs_list])
            
            trajectory.append({
                'time': t,
                'position': position,
                'velocity': velocity,
                'acceleration': acceleration
            })
            
        return trajectory