import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        # 前向传播，输出动作概率
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

# 训练策略网络的函数
def train_policy_gradient(env, policy_network, optimizer, num_episodes=1000, gamma=0.99):
    for episode in range(num_episodes):
        # 处理 env.reset() 的返回值
        state = env.reset()
        if isinstance(state, tuple):
            state, _ = state  # 忽略 info

        log_probs = []
        rewards = []
        done = False
        total_reward = 0

        while not done:
            # 确保 state 是 numpy 数组并转换为正确的形状
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            else:
                state_tensor = torch.FloatTensor([state])  # 将数值包装成列表再转换

            # 前向传播，获取动作概率分布
            action_probs = policy_network(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            # 执行动作，处理新的返回值
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward

            state = next_state

        # 计算折扣奖励
        discounted_rewards = []
        cumulative_reward = 0
        for r in rewards[::-1]:
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        # 归一化奖励
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # 计算损失函数并更新策略网络
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode+1}: Total Reward = {total_reward}")

# 初始化环境和策略网络
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]  # CartPole 环境的状态维度为 4
action_dim = env.action_space.n  # 动作维度

policy_network = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy_network.parameters(), lr=0.01)

# 开始训练
train_policy_gradient(env, policy_network, optimizer)
