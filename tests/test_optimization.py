import numpy as np
import matplotlib.pyplot as plt
import time
from src.robotic_arm_dynamics import RoboticArmDynamics, PIDController
from src.robotic_arm_optimization import MPCController, ILCController


def test_mpc_control():
    """测试非线性 MPC 控制器 (短时测试版)"""
    print("=== 开始测试 MPC 控制器 (短时测试 15 步) ===")
    arm = RoboticArmDynamics()
    mpc = MPCController(arm, horizon=2, dt=0.05)
    
    dt = 0.05
    # 【改动】：时间步只设为 15 步（对应真实时间 0.75 秒），快速看起步和收敛趋势
    time_steps = 15  
    t_span = np.arange(0, time_steps * dt, dt)
    
    q = np.zeros(6)
    q_dot = np.zeros(6)
    q_target = np.array([0.5, 0.3, -0.2, 0.0, 0.5, 0.0])
    
    q_history = []
    
    import time
    start_time = time.time()
    
    for step in range(time_steps):
        q_history.append(q.copy())
        
        tau_mpc = mpc.compute(q, q_dot, q_target)
        tau_total = tau_mpc + arm.gravity_vector(q)
        
        q_ddot = arm.forward_dynamics(q, q_dot, tau_total)
        q_dot = q_dot + q_ddot * dt
        q = q + q_dot * dt
        
        print(f"MPC 仿真进度: {step+1}/{time_steps} 步...")
            
    print(f"MPC 短时测试完成！耗时: {time.time() - start_time:.2f} 秒")
    
    # 绘图
    q_history = np.array(q_history)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(t_span, q_history[:, 1], 'b-o', linewidth=2, label='Actual J2 (MPC)')
    plt.axhline(q_target[1], color='r', linestyle='--', label='Target J2')
    plt.title('MPC Control: Early Convergence Trend')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_ilc_weekend_challenge():
    """周末挑战：通过 ILC 让机械臂学习画正弦波 (肌肉记忆养成)"""
    print("\n=== 开始周末挑战: ILC 迭代学习控制 ===")
    arm = RoboticArmDynamics()
    
    dt = 0.01
    time_steps = 200  # 2.0 秒的一段轨迹
    t_span = np.arange(0, time_steps * dt, dt)
    
    # 目标轨迹：平滑的正弦波
    q_ref = np.zeros((time_steps, 6))
    q_dot_ref = np.zeros((time_steps, 6))
    for i, t in enumerate(t_span):
        q_ref[i, 1] = 0.5 * (1 - np.cos(np.pi * t))
        q_dot_ref[i, 1] = 0.5 * np.pi * np.sin(np.pi * t)
        
    # 初始化控制器
    # 1. 兜底反馈 PD：防止在第1次迭代时机械臂掉下去
    pd_feedback = PIDController(kp=[100]*6, ki=[0]*6, kd=[10]*6)
    # 2. 前馈 ILC：专门负责把 PD 产生的“稳态/滞后误差”给学掉
    ilc = ILCController(num_points=time_steps, num_joints=6, kp=15.0, kd=1.5)
    
    num_iterations = 15
    learning_curve = []
    traj_history = []
    
    for iteration in range(num_iterations):
        q = np.zeros(6)
        q_dot = np.zeros(6)
        q_traj, err_traj, err_dot_traj = [], [], []
        
        # 执行一局完整的运动
        for step in range(time_steps):
            q_traj.append(q.copy())
            
            error = q_ref[step] - q
            error_dot = q_dot_ref[step] - q_dot
            err_traj.append(error)
            err_dot_traj.append(error_dot)
            
            # 【核心架构】：总力矩 = PD反馈 + ILC记忆前馈 + 物理重力补偿
            tau_pd = pd_feedback.compute(q_ref[step], q, q_dot, step*dt)
            tau_ilc = ilc.get_feedforward_torque(step)
            tau_gravity = arm.gravity_vector(q)
            
            tau_total = tau_pd + tau_ilc + tau_gravity
            tau_total = np.clip(tau_total, -250.0, 250.0) # 物理限幅
            
            # 步进
            q_ddot = arm.forward_dynamics(q, q_dot, tau_total)
            q_dot = q_dot + q_ddot * dt
            q = q + q_dot * dt
            
        # 一局结束，进行"复盘"更新记忆
        avg_error = ilc.update_memory(err_traj, err_dot_traj)
        learning_curve.append(avg_error)
        traj_history.append(np.array(q_traj))
        
        print(f"迭代 {iteration+1:2d}/{num_iterations} | 平均误差: {avg_error:.5f} rad")
        
    # 可视化 ILC 学习成果
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    axs[0].plot(t_span, q_ref[:, 1], 'k--', linewidth=2, label='Target Trajectory')
    axs[0].plot(t_span, traj_history[0][:, 1], 'r-', alpha=0.5, label='Iter 1 (Pure PD)')
    axs[0].plot(t_span, traj_history[3][:, 1], 'y-', alpha=0.8, label='Iter 4 (Learning)')
    axs[0].plot(t_span, traj_history[-1][:, 1], 'g-', linewidth=2, label=f'Iter {num_iterations} (Mastered)')
    axs[0].set_title('Joint 2 Tracking Evolution')
    axs[0].set_xlabel('Time (s)'); axs[0].set_ylabel('Angle (rad)')
    axs[0].legend(); axs[0].grid(True)
    
    axs[1].plot(range(1, num_iterations + 1), learning_curve, 'b-o', linewidth=2)
    axs[1].set_title('ILC Error Reduction Curve')
    axs[1].set_xlabel('Iteration Number')
    axs[1].set_ylabel('Mean Absolute Error (rad)')
    axs[1].set_yscale('log')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 先测试非线性 MPC
    test_mpc_control()
    # 再运行周末挑战的 ILC 训练
    test_ilc_weekend_challenge()