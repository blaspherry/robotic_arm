"""
6自由度机械臂：第一版
实现内容：
- DH 参数初始化
- 单节 DH 4x4 变换矩阵
- 正向运动学 forward_kinematics
- 末端位置 get_end_effector_position
- 关节限位 set_joint_angles
"""

from __future__ import annotations
import numpy as np


class RoboticArm:
    def __init__(self, num_joints: int = 6):
        """
        参数:
            num_joints: 关节数量，默认6
        """

        self.num_joints = num_joints
        self.joint_angles = np.zeros(self.num_joints, dtype=float)

        # 示例连杆长度
        # 单位：m
        self.link_lengths = [0.1, 0.3, 0.25, 0.15, 0.1, 0.05]

        # 关节限位（弧度）
        # 这里先给一个通用的 [-pi, pi]，可以按真实关节范围改
        self.joint_limits = np.array([(-np.pi, np.pi)] * self.num_joints, dtype=float)

        # DH 参数表：每行 [theta, d, a, alpha]
        self.dh_params = self._initialize_dh_parameters()

    def _initialize_dh_parameters(self) -> np.ndarray:
        """
        初始化 DH 参数表
        返回:
            dh_params: shape (num_joints, 4), 每行 [theta, d, a, alpha]
        """

        L = self.link_lengths

        dh = np.array([
            [0.0,  L[0], 0.0,  np.pi / 2],  # Joint 1
            [0.0,  0.0,  L[1], 0.0],        # Joint 2
            [0.0,  0.0,  L[2], 0.0],        # Joint 3
            [0.0,  0.0,  L[3], np.pi / 2],  # Joint 4
            [0.0,  0.0,  0.0, -np.pi / 2],  # Joint 5
            [0.0,  L[4] + L[5], 0.0, 0.0],  # Joint 6
        ], dtype=float)

        return dh

    
    def dh_transform(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """
        计算单个 DH 变换矩阵 (4x4)

        标准 DH：
            T = RotZ(theta) * TransZ(d) * TransX(a) * RotX(alpha)

        返回:
            T: 4x4 齐次变换矩阵
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        T = np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,     sa,      ca,      d],
            [0.0,    0.0,     0.0,    1.0],
        ], dtype=float)

        return T

    def forward_kinematics(self, joint_angles = None):
        """
        正向运动学：从 base 到 end-effector 的位姿

        参数:
            joint_angles: shape (num_joints,), 若为 None 则使用 self.joint_angles

        返回:
            T: 4x4 末端齐次变换矩阵
            joint_positions: list[np.ndarray], 每个关节（含末端）的 xyz 位置
        """
        if joint_angles is None:
            joint_angles = self.joint_angles

        joint_angles = np.asarray(joint_angles, dtype=float)

        T = np.eye(4, dtype=float)
        joint_positions = [T[:3, 3].copy()]  # base 原点

        for i in range(self.num_joints):
            theta, d, a, alpha = self.dh_params[i]
            theta += joint_angles[i]
            Ti = self.dh_transform(theta, d, a, alpha)
            T = T @ Ti
            joint_positions.append(T[:3, 3].copy())

        return T, joint_positions

    def get_end_effector_position(self) -> np.ndarray:
        """
        获取末端执行器位置
        返回:
            position: [x, y, z]
        """
        T, _ = self.forward_kinematics()
        return T[:3, 3].copy()

    def set_joint_angles(self, angles):
        """
        设置关节角并执行限位裁剪
        参数:
            angles: list/np.ndarray, length = num_joints
        """
        angles = np.asarray(angles, dtype=float)

        clipped = angles.copy()
        for i, (min_a, max_a) in enumerate(self.joint_limits):
            if clipped[i] < min_a or clipped[i] > max_a:
                # 超限提示 + 裁剪
                print(f"警告： Joint {i+1} 角度超出限制")
                clipped[i] = np.clip(clipped[i], min_a, max_a)

        self.joint_angles = clipped


if __name__ == "__main__":
    arm = RoboticArm(num_joints=6)

    print("测试1：零位")
    arm.set_joint_angles([0, 0, 0, 0, 0, 0])
    print("末端位置:", arm.get_end_effector_position())

    print("\n测试2：随机角度")
    arm.set_joint_angles([0.5, 0.5, -0.5, 0.0, 0.5, 0.0])
    print("末端位置:", arm.get_end_effector_position())

    print("\n测试3：性能粗测")
    import time
    num_tests = 10000
    start = time.time()
    for _ in range(num_tests):
        angles = np.random.uniform(-np.pi, np.pi, 6)
        arm.forward_kinematics(angles)
    elapsed = time.time() - start
    ops = num_tests / max(elapsed, 1e-9)
    print(f"性能: {ops:.1f} ops/sec")