import torch
import numpy as np
import matplotlib.pyplot as plt
from src.robotic_arm_dynamics import RoboticArmDynamics
from src.robotic_arm_PPO import RoboticArmEnv, PPOAgent

def train_and_test_ppo():
    arm = RoboticArmDynamics()
    env = RoboticArmEnv(arm, dt=0.02, max_steps=100)
    
    # 启用全新参数套餐：24维感知，细腻学习率，短视折扣因子
    agent = PPOAgent(state_dim=24, action_dim=6, lr=1e-4, gamma=0.90)
    
    max_episodes = 1000 
    # 扩大批次大小，积攒 2000 步(20局)经验再反思，防止被一局的运气带偏
    update_timestep = 2000 
    
    memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
    timestep = 0
    episode_rewards = []
    
    print("=== 开始 PPO 强化学习训练 (终极黑科技版) ===")
    
    for ep in range(max_episodes):
        state = env.reset()
        current_ep_reward = 0
        
        for t in range(env.max_steps):
            timestep += 1
            
            action, logprob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['logprobs'].append(logprob)
            memory['rewards'].append(reward)
            memory['is_terminals'].append(done)
            
            state = next_state
            current_ep_reward += reward
            
            if timestep % update_timestep == 0:
                agent.update(memory)
                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
                timestep = 0
                
            if done:
                break
                
        episode_rewards.append(current_ep_reward)
        if (ep + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {ep+1:4d}/{max_episodes} \t 过去20局平均得分: {avg_reward:.2f}")

    print("\n=== 验收时刻：PPO 控制机械臂到达目标并悬停 ===")
    state = env.reset()
    q_history = []
    
    agent.policy_old.eval()
    
    for _ in range(200): 
        q_history.append(env.q.copy())
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_mean = agent.policy_old.actor(state_tensor)
            action = action_mean.numpy().flatten()
            
        state, _, _ = env.step(action)
        
    q_history = np.array(q_history)
    t_span = np.arange(0, 200 * env.dt, env.dt)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    axs[0].plot(episode_rewards, color='orange', alpha=0.6)
    # 计算移动平均线，让曲线更清晰
    window = 20
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axs[0].plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2)
        
    axs[0].set_title("PPO Learning Curve")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].grid(True)
    
    axs[1].plot(t_span, q_history[:, 1], 'g-', linewidth=2, label="PPO AI Actual")
    axs[1].axhline(env.q_target[1], color='r', linestyle='--', label="Target Position")
    axs[1].set_title("Joint 2: Reaching & Hovering")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Angle (rad)")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_test_ppo()