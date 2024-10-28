import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 输出每个动作的 Q 值

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

def epsilon_greedy_policy(Q_network, state, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)  # 随机选择动作
    else:
        with torch.no_grad():
            return torch.argmax(Q_network(state)).item()  # 选择具有最大 Q 值的动作


env = gym.make('CartPole-v1')

# 获取状态和动作空间的大小
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 初始化 Q 网络和目标网络
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())  # 初始化目标网络参数为 Q 网络的参数

# 设置优化器
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

# 初始化经验回放缓冲区
replay_buffer = ReplayBuffer(10000)

# 超参数
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 64
target_update_freq = 100  # 每隔 100 步更新目标网络
num_episodes = 500

for episode in range(num_episodes):
    state,_ = env.reset()
    if isinstance(state, np.ndarray):
        # 检查 state 的长度
        print(f"State shape: {state.shape}")
        state = torch.FloatTensor(state).unsqueeze(0)  # 确保将其转化为 Tensor 并扩展维度
    else:
        print(f"Unexpected state type: {type(state)}")


    total_reward = 0
    done = False

    while not done:
        # 选择动作
        action = epsilon_greedy_policy(q_network, state, epsilon, action_dim)

        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        total_reward += reward

        # 将经验存入回放缓冲区
        replay_buffer.push((state, action, reward, next_state, done))

        # 更新状态
        state = next_state

        # 训练 Q 网络
        if replay_buffer.size() > batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.cat(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.cat(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            # 计算 Double DQN 的目标值
            next_actions = torch.argmax(q_network(next_states), dim=1, keepdim=True)
            target_q_values = target_network(next_states).gather(1, next_actions)
            target_values = rewards + gamma * target_q_values * (1 - dones)

            # 计算当前 Q 值
            q_values = q_network(states).gather(1, actions)

            # 计算损失并优化
            loss = nn.MSELoss()(q_values, target_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新目标网络
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

    # 更新 epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

