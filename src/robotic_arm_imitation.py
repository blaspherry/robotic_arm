import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.robotic_arm_dynamics import RoboticArmDynamics, ComputedTorqueController

# ==========================================
# 新增：数据归一化神器 (Data Normalizer)
# ==========================================
class DataNormalizer:
    def __init__(self):
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None

    def fit(self, states, actions):
        # 计算每一列的均值和标准差
        self.state_mean = np.mean(states, axis=0)
        # 加上 1e-8 防止除以 0 的情况
        self.state_std = np.std(states, axis=0) + 1e-8 
        
        self.action_mean = np.mean(actions, axis=0)
        self.action_std = np.std(actions, axis=0) + 1e-8

    def normalize_state(self, state):
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action):
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, norm_action):
        # 神经网络输出的是归一化后的力矩，需要还原成真实的物理力矩
        return norm_action * self.action_std + self.action_mean

# （注意：请同时修改 BehavioralCloningAgent 的 train 方法，加入学习率衰减）
# 找到你的 BehavioralCloningAgent，把 __init__ 和 train 改成这样：
class BehavioralCloningAgent:
    def __init__(self, state_dim=30, action_dim=6, hidden_dim=256, lr=1e-3):
        self.policy = ImitationPolicy(state_dim, action_dim, hidden_dim)
        # 加上 weight_decay (L2正则化) 防止网络死记硬背
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        # 新增：学习率调度器，当 Loss 不再下降时，自动缩小学习率进行精细微调
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5)
        
    def train(self, dataset, epochs=50, batch_size=128):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        
        print(f"=== 开启神经网络训练 (共 {epochs} 轮) ===")
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_states, batch_actions in dataloader:
                predicted_actions = self.policy(batch_states)
                loss = self.criterion(predicted_actions, batch_actions)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # 步进调度器
            self.scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{epochs}], MSE Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
                
        return losses
        
    def predict(self, state):
        self.policy.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.policy(state_tensor).squeeze(0).numpy()
        return action

# ==========================================
# 1. TODO: 实现轨迹数据集 (PyTorch Dataset)
# ==========================================
class ExpertDataset(Dataset):
    def __init__(self, states, actions):
        """
        接收收集好的专家数据
        states: 维度 (N, 12) -> [q1..q6, qdot1..qdot6]
        actions: 维度 (N, 6) -> [tau1..tau6]
        """
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# ==========================================
# 2. TODO: 构建神经网络策略 (Policy Network)
# ==========================================
class ImitationPolicy(nn.Module):
    def __init__(self, state_dim=12, action_dim=6, hidden_dim=256):
        """
        构建一个多层感知机 (MLP) 大脑 

[Image of Artificial neural network architecture]

        """
        super(ImitationPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # 输出层不需要激活函数，因为力矩可以是正负任意值
        )
        
    def forward(self, state):
        return self.network(state)

# ==========================================
# 3. TODO: 实现行为克隆智能体 (Behavioral Cloning Agent)
# ==========================================
class BehavioralCloningAgent:
    def __init__(self, state_dim=12, action_dim=6, hidden_dim=256, lr=1e-3):
        self.policy = ImitationPolicy(state_dim, action_dim, hidden_dim)
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # 损失函数：均方误差 (MSE)，让网络输出的力矩无限逼近专家力矩
        self.criterion = nn.MSELoss()
        
    def train(self, dataset, epochs=50, batch_size=128):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        
        print(f"=== 开启神经网络训练 (共 {epochs} 轮) ===")
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_states, batch_actions in dataloader:
                # 前向传播：网络猜一个力矩
                predicted_actions = self.policy(batch_states)
                
                # 计算与老司机的误差
                loss = self.criterion(predicted_actions, batch_actions)
                
                # 反向传播更新权重
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], MSE Loss: {avg_loss:.6f}")
                
        return losses
        
    def predict(self, state):
        """测试阶段：输入当前状态，网络直接凭借直觉输出力矩"""
        self.policy.eval() # 切换到评估模式
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.policy(state_tensor).squeeze(0).numpy()
        return action

# ==========================================
# 4. 数据收集器 (升级版：消除信息不对称)
# ==========================================
def collect_expert_data(arm, num_trajectories=30, time_steps=200, dt=0.01):
    expert = ComputedTorqueController(arm, kp=[100]*6, kd=[20]*6)
    states, actions = [], []
    print(f"老司机 CTC 正在跑 {num_trajectories} 趟车收集数据...")
    
    for _ in range(num_trajectories):
        freq = np.random.uniform(0.5, 1.5)
        amp = np.random.uniform(0.2, 0.6)
        
        q = np.zeros(6)
        q_dot = np.zeros(6)
        
        for step in range(time_steps):
            t = step * dt
            
            # 生成目标点 (AI 需要知道要去哪)
            q_d = amp * (1 - np.cos(np.pi * freq * t)) * np.ones(6)
            q_dot_d = amp * np.pi * freq * np.sin(np.pi * freq * t) * np.ones(6)
            q_ddot_d = amp * (np.pi * freq)**2 * np.cos(np.pi * freq * t) * np.ones(6)
            
            # 【改进 1：真正的救车教学】
            # 先给当前真实物理状态加上噪音，模拟 AI 跑偏的情况
            noisy_q = q + np.random.normal(0, 0.005, 6)
            noisy_qdot = q_dot + np.random.normal(0, 0.02, 6)
            
            # 【极其重要】：让老司机基于这个“跑偏”的状态，计算如何“救车”的力矩！
            tau_expert_recovery = expert.compute(q_d, q_dot_d, q_ddot_d, noisy_q, noisy_qdot)
            tau_expert_recovery = np.clip(tau_expert_recovery, -300.0, 300.0)
            
            # 【改进 2：输入维度扩展到 30 维】
            # AI 的视野 = [当前位置, 当前速度, 目标位置, 目标速度, 目标加速度]
            full_state = np.concatenate([noisy_q, noisy_qdot, q_d, q_dot_d, q_ddot_d])
            
            states.append(full_state)
            actions.append(tau_expert_recovery)
            
            # 物理引擎步进 (为了轨迹连贯，依然使用老司机基于干净状态算出的完美力矩来推演)
            tau_perfect = expert.compute(q_d, q_dot_d, q_ddot_d, q, q_dot)
            tau_perfect = np.clip(tau_perfect, -300.0, 300.0)
            q_ddot = arm.forward_dynamics(q, q_dot, tau_perfect)
            q_dot = np.clip(q_dot + q_ddot * dt, -50.0, 50.0)
            q = q + q_dot * dt
            
    return np.array(states), np.array(actions)