import numpy as np
import time

class RoboticArm:
    def __init__(self, num_joints=6):
        """初始化机械臂 [cite: 82]"""
        self.num_joints = num_joints
        self.joint_angles = np.zeros(num_joints)
        
        # 连杆长度(米) [cite: 90, 91]
        self.link_lengths = [0.1, 0.3, 0.25, 0.15, 0.1, 0.05]
        
        # 关节限制(弧度) [cite: 92, 93]
        self.joint_limits = [(-np.pi, np.pi) for _ in range(6)]
        
        # 初始化DH参数表
        self.dh_params = self._initialize_dh_parameters()

    def _initialize_dh_parameters(self):
        """
        TODO完成: 定义标准的6自由度机械臂DH参数表 (类似Puma/UR构型)
        格式: [theta, d, a, alpha] for each joint [cite: 88, 111]
        """
        L = self.link_lengths
        # 根据给定的连杆长度，设计一个标准的拟人臂构型
        dh_params = np.array([
            [0, L[0], 0, np.pi/2],      # Joint 1: Base yaw
            [0, 0, L[1], 0],            # Joint 2: Shoulder pitch
            [0, 0, L[2], 0],            # Joint 3: Elbow pitch
            [0, L[3], 0, np.pi/2],      # Joint 4: Wrist roll
            [0, 0, 0, -np.pi/2],        # Joint 5: Wrist pitch
            [0, L[4] + L[5], 0, 0]      # Joint 6: Wrist roll (combining last two links to end effector)
        ])
        return dh_params

    def dh_transform(self, theta, d, a, alpha):
        """计算单个DH变换矩阵 [cite: 133]"""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        T = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [ 0,     sa,     ca,    d],
            [ 0,      0,      0,    1]
        ])
        return T

    def forward_kinematics(self, joint_angles=None):
        """TODO完成: 正向运动学计算 [cite: 170, 176]"""
        if joint_angles is None:
            joint_angles = self.joint_angles
            
        T = np.eye(4)
        joint_positions = [T[:3, 3].copy()] # 记录基座位置
        
        for i in range(self.num_joints):
            theta_offset, d, a, alpha = self.dh_params[i]
            theta = theta_offset + joint_angles[i]
            T_i = self.dh_transform(theta, d, a, alpha)
            T = T @ T_i # 累积变换 [cite: 191]
            joint_positions.append(T[:3, 3].copy())
            
        return T, joint_positions

    def get_end_effector_position(self):
        """获取末端执行器位置 [cite: 196]"""
        T, _ = self.forward_kinematics()
        return T[:3, 3]

    def set_joint_angles(self, angles):
        """设置关节角度并检查限制 [cite: 203]"""
        angles = np.array(angles)
        for i, (angle, (min_angle, max_angle)) in enumerate(zip(angles, self.joint_limits)):
            if angle < min_angle or angle > max_angle:
                angles[i] = np.clip(angle, min_angle, max_angle)
        self.joint_angles = angles

    def jacobian(self, joint_angles=None, epsilon=1e-6):
        """TODO完成: 实现数值微分计算雅可比矩阵 """
        if joint_angles is None:
            joint_angles = self.joint_angles.copy()
        else:
            joint_angles = np.array(joint_angles).copy()
            
        J = np.zeros((3, self.num_joints))
        current_pos = self.get_end_effector_position()
        
        for i in range(self.num_joints):
            original_angle = joint_angles[i]
            
            # 正向扰动
            joint_angles[i] = original_angle + epsilon
            self.joint_angles = joint_angles
            pos_plus = self.get_end_effector_position()
            
            # 负向扰动
            joint_angles[i] = original_angle - epsilon
            self.joint_angles = joint_angles
            pos_minus = self.get_end_effector_position()
            
            # 中心差分法计算偏导数
            J[:, i] = (pos_plus - pos_minus) / (2 * epsilon)
            
            # 恢复原始角度
            joint_angles[i] = original_angle
            self.joint_angles = joint_angles
            
        return J

    def inverse_kinematics(self, target_position, initial_guess=None, max_iterations=100, tolerance=1e-3, learning_rate=0.5):
        """TODO完成: 基于雅可比矩阵的逆运动学求解 """
        target_position = np.array(target_position)
        
        if initial_guess is None:
            theta = self.joint_angles.copy()
        else:
            theta = np.array(initial_guess).copy()
            
        for iteration in range(max_iterations):
            self.set_joint_angles(theta)
            current_pos = self.get_end_effector_position()
            error = target_position - current_pos
            
            if np.linalg.norm(error) < tolerance:
                return True, theta
                
            J = self.jacobian(theta)
            
            try:
                # 使用伪逆求解 
                J_pinv = np.linalg.pinv(J)
                delta_theta = learning_rate * (J_pinv @ error)
            except np.linalg.LinAlgError:
                # 阻尼最小二乘法 (处理奇异点)
                damping = 0.01
                J_damped = J.T @ J + damping * np.eye(self.num_joints)
                delta_theta = learning_rate * (np.linalg.inv(J_damped) @ J.T @ error)
                
            theta += delta_theta
            
            # 应用限制
            theta = np.clip(theta, [limit[0] for limit in self.joint_limits], [limit[1] for limit in self.joint_limits])
            
        return False, theta
    
