import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ==========================================
# 1. 强化学习环境包装器 (OpenAI Gym 风格)
# ==========================================
class RoboticArmEnv:
    def __init__(self, arm_dynamics, dt=0.02, max_steps=100):
        self.arm = arm_dynamics
        self.dt = dt
        self.max_steps = max_steps
        self.current_step = 0
        
        self.max_torque = 100.0 
        self.max_velocity = 10.0 
        
        self.q = np.zeros(6)
        self.q_dot = np.zeros(6)
        self.q_target = np.array([0.5, 0.5, -0.5, 0.2, 0.5, 0.0])
        
        # 【新增】：记录上一步动作，用于计算动作平滑惩罚
        self.last_action = np.zeros(6)

    def reset(self):
        self.current_step = 0
        self.q = np.random.uniform(-0.1, 0.1, 6)
        self.q_dot = np.zeros(6)
        self.last_action = np.zeros(6) # 回合重置时也要清零
        return self._get_state()

    def _get_state(self):
        norm_q_dot = self.q_dot / self.max_velocity
        # 【算法核心优化 1】：直接算出误差，帮神经网络分担计算压力
        error = self.q_target - self.q
        # 状态维度升级为: 6(位置) + 6(速度) + 6(目标) + 6(误差) = 24 维
        return np.concatenate([self.q, norm_q_dot, self.q_target, error])

    def step(self, action):
        self.current_step += 1
        action = np.array(action).flatten()
        
        # 1. 残差控制
        tau_gravity = self.arm.gravity_vector(self.q)
        tau_ai = action * 30.0 
        tau_total = tau_gravity + tau_ai
        
        # 2. 物理推演
        q_ddot = self.arm.forward_dynamics(self.q, self.q_dot, tau_total)
        self.q_dot = self.q_dot + q_ddot * self.dt
        self.q = self.q + self.q_dot * self.dt
        
        # 3. 幽灵动量消除
        for i in range(6):
            if self.q[i] >= np.pi:
                self.q[i] = np.pi; self.q_dot[i] = 0.0  
            elif self.q[i] <= -np.pi:
                self.q[i] = -np.pi; self.q_dot[i] = 0.0  
                
        self.q_dot = np.clip(self.q_dot, -self.max_velocity, self.max_velocity)
        
        # ==========================================
        # 计算奖励
        # ==========================================
        distance = np.linalg.norm(self.q_target - self.q)
        reward = -distance 
        
        action_diff = np.linalg.norm(action - self.last_action)
        reward -= 0.05 * action_diff
        self.last_action = np.copy(action)
        
        # ==========================================
        # 【修改点 2：高斯引力井 (Gaussian Attraction Well)】
        # 删掉原来的 if 阶梯，换成这个平滑的指数公式。
        # 效果：距离越近，吸力越强，分数呈完美抛物线平滑上升，最高直达 +15 分！
        # 这种处处可导的奖励函数，会让神经网络学得爽到飞起。
        # ==========================================
        attraction_bonus = 15.0 * np.exp(-5.0 * distance)
        reward += attraction_bonus
            
        action_penalty = 0.001 * np.linalg.norm(action)
        reward -= action_penalty
            
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done

# ==========================================
# 2. PPO 算法核心组件
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim=24, action_dim=6, hidden_dim=256): # state_dim 默认设为 24
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() 
        )
        
        self.action_log_std = nn.Parameter(torch.full((1, action_dim), -1.0))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 【算法核心优化 3】：触发 PPO 正交初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()
        # 将 Actor 最后一层权重压得极小，保证初期不乱挥
        if module == self.actor[4]:
            nn.init.orthogonal_(module.weight, gain=0.01)
        
    def act(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_logprob.detach()
        
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std)
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim=24, action_dim=6, lr=1e-4, gamma=0.90, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state)
        return action.numpy().flatten(), action_logprob.item()

    def update(self, memory):
        old_states = torch.FloatTensor(np.array(memory['states']))
        old_actions = torch.FloatTensor(np.array(memory['actions']))
        old_logprobs = torch.FloatTensor(np.array(memory['logprobs']))
        
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory['rewards']), reversed(memory['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 【调参】：大脑反思次数从 4 次提升到 10 次，榨干数据价值
        for _ in range(10): 
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            advantages = rewards - state_values.detach()
            
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2).mean() + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())