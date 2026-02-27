import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 假设前几周的 RoboticArm 类在 robotic_arm 模块中
# 如果你的文件路径不同，请自行调整这里的 import
from src.robotic_arm import RoboticArm

class RoboticArmDynamics(RoboticArm):
    def __init__(self, num_joints=6):
        """初始化动力学模型 (基于 PDF 数据)"""
        super().__init__(num_joints)
        
        # 连杆质量 (kg)
        self.link_masses = [2.0, 3.0, 2.5, 1.5, 1.0, 0.5]
        
        # 连杆质心位置 (相对于连杆坐标系)
        self.link_com_offsets = [
            [0, 0, self.link_lengths[0]/2],
            [self.link_lengths[1]/2, 0, 0],
            [self.link_lengths[2]/2, 0, 0],
            [0, 0, self.link_lengths[3]/2],
            [0, 0, 0],
            [0, 0, self.link_lengths[4]/2]
        ]
        
        # 连杆惯性张量
        self.link_inertias = []
        for i, (m, l) in enumerate(zip(self.link_masses, self.link_lengths)):
            I = (1.0/12.0) * m * l**2
            # 【防爆修复1】：增加微小的基础惯量，防止纯旋转时产生奇异矩阵
            self.link_inertias.append(np.diag([I, I, I]) + np.eye(3) * 0.001)
            
        self.gravity = np.array([0, 0, -9.81])
        self.friction_coeffs = np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])

    # --- 运动学辅助函数 ---
    def _get_link_transform(self, q, link_idx):
        """获取第 link_idx 个连杆相对于基座的变换矩阵 T"""
        T = np.eye(4)
        for i in range(link_idx + 1):
            theta, d, a, alpha = self.dh_params[i]
            theta += q[i]
            T_i = self.dh_transform(theta, d, a, alpha)
            T = T @ T_i
        return T

    def _calculate_com_position(self, q, link_idx):
        """计算特定连杆质心的全局位置"""
        T = self._get_link_transform(q, link_idx)
        com_local = np.append(self.link_com_offsets[link_idx], 1)
        return (T @ com_local)[:3]

    def _com_jacobian(self, q, link_idx):
        """计算质心线速度雅可比矩阵 (数值微分法)"""
        J_v = np.zeros((3, self.num_joints))
        epsilon = 1e-5
        for i in range(self.num_joints):
            if i > link_idx: continue
            q_plus, q_minus = q.copy(), q.copy()
            q_plus[i] += epsilon
            q_minus[i] -= epsilon
            pos_plus = self._calculate_com_position(q_plus, link_idx)
            pos_minus = self._calculate_com_position(q_minus, link_idx)
            J_v[:, i] = (pos_plus - pos_minus) / (2 * epsilon)
        return J_v

    def _angular_velocity_jacobian(self, q, link_idx):
        """【防爆修复2】：计算角速度雅可比，彻底消除奇异矩阵"""
        J_w = np.zeros((3, self.num_joints))
        T = np.eye(4)
        J_w[:, 0] = np.array([0, 0, 1]) 
        for i in range(1, link_idx + 1):
            theta, d, a, alpha = self.dh_params[i-1]
            theta += q[i-1]
            T = T @ self.dh_transform(theta, d, a, alpha)
            if i <= link_idx:
                J_w[:, i] = T[:3, 2]
        return J_w

    # --- 动力学核心函数 ---
    def mass_matrix(self, q):
        """计算惯性矩阵 M(q)"""
        M = np.zeros((self.num_joints, self.num_joints))
        for k in range(self.num_joints):
            J_v = self._com_jacobian(q, k)
            J_w = self._angular_velocity_jacobian(q, k)
            R_k = self._get_link_transform(q, k)[:3, :3]
            I_k_global = R_k @ self.link_inertias[k] @ R_k.T
            
            # 平移和旋转动能
            M += self.link_masses[k] * (J_v.T @ J_v) + (J_w.T @ I_k_global @ J_w)
            
        M = (M + M.T) / 2.0
        # 【防爆修复3】：给对角线添加物理电机转子的微小惯量，绝对保证正定
        M += np.eye(self.num_joints) * 0.05
        return M

    def coriolis_matrix(self, q, q_dot):
        """计算科里奥利矩阵 C"""
        C = np.zeros((self.num_joints, self.num_joints))
        epsilon = 1e-5
        M_curr = self.mass_matrix(q)
        dM_dq = np.zeros((self.num_joints, self.num_joints, self.num_joints))
        
        for k in range(self.num_joints):
            q_plus = q.copy()
            q_plus[k] += epsilon
            dM_dq[:, :, k] = (self.mass_matrix(q_plus) - M_curr) / epsilon
            
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                c_ij = 0
                for k in range(self.num_joints):
                    c_ij += 0.5 * (dM_dq[i, j, k] + dM_dq[i, k, j] - dM_dq[j, k, i]) * q_dot[k]
                C[i, j] = c_ij
        return C

    def gravity_vector(self, q):
        """计算重力项 G(q)"""
        G = np.zeros(self.num_joints)
        for k in range(self.num_joints):
            J_v = self._com_jacobian(q, k)
            G -= J_v.T @ (self.link_masses[k] * self.gravity)
        return G

    def forward_dynamics(self, q, q_dot, tau):
        """正向动力学计算"""
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)
        friction = self.friction_coeffs * q_dot
        
        bias = tau - (C @ q_dot) - G - friction
        
        try:
            q_ddot = np.linalg.solve(M, bias)
        except np.linalg.LinAlgError:
            q_ddot = np.linalg.pinv(M) @ bias
            
        # 【防爆修复4】：加速度限幅，彻底杜绝数据变成 1e+36
        q_ddot = np.clip(q_ddot, -1000.0, 1000.0)
        return q_ddot

    def inverse_dynamics(self, q, q_dot, q_ddot):
        """逆向动力学计算"""
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)
        friction = self.friction_coeffs * q_dot
        return M @ q_ddot + C @ q_dot + G + friction

    def simulate(self, q0, q_dot0, tau_func, t_span, dt=0.01):
        """使用 odeint 进行动力学仿真"""
        t_eval = np.arange(t_span[0], t_span[1], dt)
        
        def dynamics(state, t):
            q = state[:self.num_joints]
            q_dot = state[self.num_joints:]
            tau = tau_func(t, q, q_dot)
            q_ddot = self.forward_dynamics(q, q_dot, tau)
            return np.concatenate([q_dot, q_ddot])
            
        state0 = np.concatenate([q0, q_dot0])
        states = odeint(dynamics, state0, t_eval)
        return t_eval, states[:, :self.num_joints], states[:, self.num_joints:]


# ==========================================
# 控制器模块
# ==========================================

class PIDController:
    def __init__(self, kp, ki, kd):
        # 【防爆修复5】：强制转换 dtype=float，防止 float 存入 int 报错
        self.kp = np.array(kp, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.integral = np.zeros_like(self.kp)
        self.last_t = 0.0

    def compute(self, q_target, q_current, q_dot_current, t=0.0):
        error = q_target - q_current
        
        # 【防爆修复6】：防止 odeint 的微步试探导致积分项爆炸
        dt = t - self.last_t
        if dt > 0 and dt < 0.1: 
            self.integral += error * dt
            self.last_t = t
            
        tau = self.kp * error + self.ki * self.integral - self.kd * q_dot_current
        # 【防爆修复7】：力矩截断保护
        tau = np.clip(tau, -300.0, 300.0)
        return tau

class ComputedTorqueController:
    def __init__(self, arm_dynamics, kp, kd):
        self.arm = arm_dynamics
        self.kp = np.array(kp, dtype=float)
        self.kd = np.array(kd, dtype=float)

    def compute(self, q_d, q_dot_d, q_ddot_d, q, q_dot):
        e = q_d - q
        e_dot = q_dot_d - q_dot
        aq = q_ddot_d + self.kp * e + self.kd * e_dot
        tau = self.arm.inverse_dynamics(q, q_dot, aq)
        tau = np.clip(tau, -300.0, 300.0)
        return tau


# ==========================================
# 测试代码
# ==========================================

def test_pid_control():
    print("=== 测试PID控制 ===")
    arm = RoboticArmDynamics(num_joints=6)
    
    # 调柔参数，保证阶跃响应绝对不爆炸
    kp = [100, 100, 100, 50, 50, 50]
    ki = [0, 0, 0, 0, 0, 0] # 阶跃测试中保持 I 为 0 最稳
    kd = [15, 15, 15, 5, 5, 5]
    
    controller = PIDController(kp, ki, kd)
    
    q0 = np.zeros(6)
    q_dot0 = np.zeros(6)
    q_target = np.array([0.5, 0.5, -0.5, 0.2, 0.5, 0])
    
    # 传入 t 到控制律中
    def control_law(t, q, q_dot):
        # 1. 计算原本的 PD 反馈力矩 (此时可以把积分项 Ki 继续保持为 0，非常安全)
        tau_pd = controller.compute(q_target, q, q_dot, t)
        
        # 2. 【核心魔法】：调用物理引擎，算出当前姿态下，重力把各个关节往下拽的力矩是多少
        tau_gravity = arm.gravity_vector(q)
        
        # 3. 总输出力矩 = PD用来消除动态误差 + 直接抵消重力
        return tau_pd + tau_gravity
    
    t, q_hist, _ = arm.simulate(q0, q_dot0, control_law, [0, 3.0])
    
    plt.figure(figsize=(12, 8))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(t, q_hist[:, i], label='Actual')
        plt.axhline(q_target[i], color='r', linestyle='--', label='Target')
        plt.title(f'Joint {i+1}')
        plt.xlabel('Time (s)')
        plt.grid(True)
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.show()

def compare_controllers():
    print("=== 测试计算力矩控制 (CTC) ===")
    arm = RoboticArmDynamics(num_joints=6)
    kp = [100]*6
    kd = [20]*6
    ctc = ComputedTorqueController(arm, kp, kd)
    
    q0 = np.zeros(6)
    q_dot0 = np.zeros(6)
    
    def get_target(t):
        freq = 1.0
        q_d = 0.3 * np.sin(freq * t) * np.ones(6)
        q_dot_d = 0.3 * freq * np.cos(freq * t) * np.ones(6)
        q_ddot_d = -0.3 * freq**2 * np.sin(freq * t) * np.ones(6)
        return q_d, q_dot_d, q_ddot_d
        
    def control_law(t, q, q_dot):
        q_d, dq_d, ddq_d = get_target(t)
        return ctc.compute(q_d, dq_d, ddq_d, q, q_dot)
        
    t, q_hist, _ = arm.simulate(q0, q_dot0, control_law, [0, 3.0])
    
    plt.figure(figsize=(8, 5))
    plt.plot(t, q_hist[:, 0], 'g-', linewidth=2, label="CTC Actual J1")
    plt.plot(t, 0.3 * np.sin(t), 'k--', linewidth=2, label="Target J1")
    plt.legend()
    plt.title("Computed Torque Control Tracking Performance")
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test_pid_control()
    compare_controllers()