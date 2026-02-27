import numpy as np
import matplotlib.pyplot as plt
from src.robotic_arm_dynamics import RoboticArmDynamics
from src.robotic_arm_imitation import ExpertDataset, BehavioralCloningAgent, collect_expert_data, DataNormalizer

def test_behavioral_cloning():
    arm = RoboticArmDynamics()
    dt = 0.01
    time_steps = 200
    
    # 1. 扩大一点数据量，收集 60 趟车的数据
    states, actions = collect_expert_data(arm, num_trajectories=60, time_steps=time_steps, dt=dt)
    
    # ==========================================
    # 核心优化：训练归一化器
    # ==========================================
    normalizer = DataNormalizer()
    normalizer.fit(states, actions)
    
    norm_states = normalizer.normalize_state(states)
    norm_actions = normalizer.normalize_action(actions)
    
    # 2. 封装归一化后的数据并训练
    dataset = ExpertDataset(norm_states, norm_actions)
    agent = BehavioralCloningAgent(state_dim=30, action_dim=6, hidden_dim=256, lr=0.001)
    
    # 此时的 Loss 是在归一化空间计算的，数值会极其小 (比如 0.05)
    losses = agent.train(dataset, epochs=100, batch_size=256) 
    
    # 3. 周末挑战测试
    print("=== 周末挑战：神经网络独立控制测试 ===")
    t_span = np.arange(0, time_steps * dt, dt)
    q_nn = np.zeros(6)
    q_dot_nn = np.zeros(6)
    q_history_nn = []
    
    for step in range(time_steps):
        t = step * dt
        q_history_nn.append(q_nn.copy())
        
        q_d = np.array([0.4 * (1 - np.cos(2.0 * np.pi * t)) + 0.1 * np.sin(4.0 * np.pi * t)] * 6)
        q_dot_d = np.array([0.4 * 2.0 * np.pi * np.sin(2.0 * np.pi * t) + 0.4 * np.pi * np.cos(4.0 * np.pi * t)] * 6)
        q_ddot_d = np.array([0.4 * (2.0 * np.pi)**2 * np.cos(2.0 * np.pi * t) - 1.6 * np.pi**2 * np.sin(4.0 * np.pi * t)] * 6)
        
        current_state = np.concatenate([q_nn, q_dot_nn, q_d, q_dot_d, q_ddot_d])
        
        # ==========================================
        # 核心优化：预测时的归一化与反归一化
        # ==========================================
        # 先把当前物理状态压缩成网络喜欢的形状
        norm_state = normalizer.normalize_state(current_state)
        
        # 网络预测出归一化后的力矩
        norm_action_pred = agent.predict(norm_state)
        
        # 把网络输出还原成真实的物理力矩
        tau_pred = normalizer.denormalize_action(norm_action_pred)
        
        # 护航物理引擎
        tau_pred = np.clip(tau_pred, -300.0, 300.0)
        
        q_ddot = arm.forward_dynamics(q_nn, q_dot_nn, tau_pred)
        q_dot_nn = np.clip(q_dot_nn + q_ddot * dt, -50.0, 50.0)
        q_nn = q_nn + q_dot_nn * dt

    q_history_nn = np.array(q_history_nn)
    
    # 4. 绘图代码保持不变...
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].plot(losses, 'm-', linewidth=2)
    axs[0].set_title("Training Loss (Normalized Space)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("MSE Loss")
    axs[0].set_yscale('log')
    axs[0].grid(True)
    
    target_j2 = [0.4 * (1 - np.cos(2.0 * np.pi * t)) + 0.1 * np.sin(4.0 * np.pi * t) for t in t_span]
    axs[1].plot(t_span, target_j2, 'k--', linewidth=2, label="Target Trajectory")
    axs[1].plot(t_span, q_history_nn[:, 1], 'g-', linewidth=2, label="Neural Network Policy")
    axs[1].set_title("Joint 2 Tracking (With Normalization)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Angle (rad)")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_behavioral_cloning()