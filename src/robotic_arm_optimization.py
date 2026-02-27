import numpy as np
from scipy.optimize import minimize
from src.robotic_arm_dynamics import RoboticArmDynamics, PIDController

class MPCController:
    def __init__(self, arm_dynamics, horizon=3, dt=0.05):
        """
        TODO: 实现模型预测控制器 (MPC)
        通过求解有限时域内的二次规划问题，得到最优控制序列。
        """
        self.arm = arm_dynamics
        self.horizon = horizon  # 预测步数 (N)
        self.dt = dt            # 预测步长
        self.num_joints = arm_dynamics.num_joints
        
        # 权重矩阵: Q惩罚跟踪误差，R惩罚控制能量消耗
        self.Q = np.diag([100.0] * self.num_joints)
        self.R = np.diag([0.1] * self.num_joints)
        
        # 物理限制: 电机最大扭矩
        self.u_bound = 150.0

    def compute(self, q_current, q_dot_current, q_target):
        """计算当前时刻的最优力矩"""
        # 初始猜测：未来 horizon 步的力矩全部为 0
        u_init = np.zeros(self.horizon * self.num_joints)
        
        # 定义代价函数 J = sum(e^T Q e + u^T R u)
        def cost_function(u_flat):
            u_seq = u_flat.reshape(self.horizon, self.num_joints)
            cost = 0
            
            # 在大脑中推演未来状态 (由于是在优化器内，为了速度使用简单欧拉法)
            q_pred = q_current.copy()
            q_dot_pred = q_dot_current.copy()
            
            for k in range(self.horizon):
                u_k = u_seq[k]
                
                # 调用第5周的动力学引擎推演未来
                q_ddot = self.arm.forward_dynamics(q_pred, q_dot_pred, u_k)
                q_dot_pred = q_dot_pred + q_ddot * self.dt
                q_pred = q_pred + q_dot_pred * self.dt
                
                # 计算当前预测步的代价
                error = q_target - q_pred
                cost += np.dot(error.T, np.dot(self.Q, error)) + np.dot(u_k.T, np.dot(self.R, u_k))
                
            return cost

        # 设定力矩的上下界约束
        bounds = [(-self.u_bound, self.u_bound) for _ in range(self.horizon * self.num_joints)]
        
        # 使用 SLSQP 求解带有约束的非线性优化问题
        # 注意：真实工业中会使用 C++ 编写的 OSQP/ACADO，这里用 scipy 限制迭代次数防卡死
        result = minimize(
            cost_function, 
            u_init, 
            method='SLSQP', 
            bounds=bounds,
            options={'maxiter': 10, 'disp': False} 
        )
        
        # 滚动优化核心思想：只取最优序列中的第一个动作执行
        u_optimal = result.x[:self.num_joints]
        return u_optimal


class ILCController:
    def __init__(self, num_points, num_joints=6, kp=15.0, kd=1.5):
        """
        TODO: 实现迭代学习控制器 (ILC)
        基于前馈记忆的 PD 型学习律，通过反复练习消除重复轨迹的稳态误差。
        """
        self.num_points = num_points
        self.num_joints = num_joints
        
        # 跨迭代的学习率 (不同于底层的反馈PD)
        self.kp = kp
        self.kd = kd
        
        # 核心记忆体：存储整条轨迹每个时间步的前馈力矩
        self.u_memory = np.zeros((num_points, num_joints))
        
    def get_feedforward_torque(self, step):
        """在当前迭代执行时，提取记忆中的前馈力矩"""
        if step < self.num_points:
            return self.u_memory[step]
        return self.u_memory[-1]

    def update_memory(self, error_history, error_dot_history):
        """
        在一次完整的运动结束后，进行“复盘”并更新记忆
        公式: u_{k+1} = u_k + Kp * e + Kd * de
        """
        error_history = np.array(error_history)
        error_dot_history = np.array(error_dot_history)
        
        # 【黑科技】：超前相位学习 (Phase-Lead)
        # 将误差数组整体向前移动 1 个步长，抵消机械臂的物理惯性延迟，极大加速收敛
        e_lead = np.roll(error_history, -1, axis=0)
        e_lead[-1] = error_history[-1]
        
        edot_lead = np.roll(error_dot_history, -1, axis=0)
        edot_lead[-1] = error_dot_history[-1]
        
        # 更新大脑中的力矩记忆
        self.u_memory += self.kp * e_lead + self.kd * edot_lead
        
        # 安全护航：记忆力矩绝对不能超过电机极限，防止下一次迭代直接爆炸
        self.u_memory = np.clip(self.u_memory, -200.0, 200.0)
        
        # 返回平均绝对误差，用于绘制学习曲线
        return np.mean(np.abs(error_history))