import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        # 前向传播，输出每个动作的 Q 值
        x = torch.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# 训练 DQN 的函数
def train_dqn(env, q_network, optimizer, num_episodes=500, gamma=0.99, batch_size=64):
    replay_buffer = ReplayBuffer()
    criterion = nn.MSELoss()
    epsilon = 1.0  # 初始探索率
    epsilon_decay = 0.995
    min_epsilon = 0.01

    for episode in range(num_episodes):
        # 处理 env.reset() 的返回值
        state, info = env.reset()
        # 确保 state 是 NumPy 数组
        state = np.array(state)
        total_reward = 0
        done = False

        while not done:
            # ε-贪心策略选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values).item()

            # 执行动作，处理新的返回值
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # 确保 next_state 是 NumPy 数组
            next_state = np.array(next_state)

            # 存储转移样本
            replay_buffer.push((state, action, reward, next_state, done))
            state = next_state

            # 当经验池足够大时开始训练
            if len(replay_buffer.buffer) >= batch_size:
                transitions = replay_buffer.sample(batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

                # 将 batch_state 转换为 NumPy 数组并检查形状
                batch_state = np.array(batch_state)
                batch_next_state = np.array(batch_next_state)

                # 检查状态维度是否正确
                if batch_state.shape[1] != state_dim:
                    print(f"Error: Expected state dimension {state_dim}, but got {batch_state.shape[1]}")
                    continue  # 跳过这次训练，继续下一个 batch

                batch_state = torch.FloatTensor(batch_state)
                batch_action = torch.LongTensor(batch_action).unsqueeze(1)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
                batch_next_state = torch.FloatTensor(batch_next_state)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1)

                # 计算当前 Q 值
                current_q_values = q_network(batch_state).gather(1, batch_action)

                # 计算下一个状态的最大 Q 值
                next_q_values = q_network(batch_next_state).max(1)[0].unsqueeze(1)
                target_q_values = batch_reward + gamma * next_q_values * (1 - batch_done)

                # 计算损失并更新网络
                loss = criterion(current_q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode+1}: Total Reward = {total_reward}")

# 初始化环境和 Q 网络
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_network = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

# 开始训练
train_dqn(env, q_network, optimizer)
