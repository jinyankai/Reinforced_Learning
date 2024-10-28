import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import random
from collections import deque


# 定义 Dueling Q 网络结构，包含状态价值分支 V(s) 和优势分支 A(s, a)
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()

        # 公共层：输入状态 -> 两个隐层
        self.fc1 = nn.Linear(state_dim, 128)  # 第一个全连接层，输入状态维度
        self.fc2 = nn.Linear(128, 128)  # 第二个全连接层

        # 状态价值分支：输出为一个值，代表 V(s)
        self.value_fc = nn.Linear(128, 1)

        # 优势分支：输出为每个动作的优势值 A(s, a)
        self.advantage_fc = nn.Linear(128, action_dim)

    # 前向传播，输出 Q 值
    def forward(self, state):
        x = torch.relu(self.fc1(state))  # 输入状态经过第一个隐藏层
        x = torch.relu(self.fc2(x))  # 经过第二个隐藏层

        # 计算状态价值 V(s) 和优势 A(s, a)
        value = self.value_fc(x)  # 状态价值 V(s)
        advantage = self.advantage_fc(x)  # 优势 A(s, a)

        # Q(s, a) = V(s) + A(s, a) - mean(A(s, a))，确保优势函数的均值为 0
        Q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return Q


# 定义经验回放缓冲区，用于存储与环境交互的经验
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 使用 deque 来存储经验，容量有限

    # 添加经验到缓冲区
    def push(self, experience):
        self.buffer.append(experience)

    # 从缓冲区随机采样 batch_size 大小的经验
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    # 获取缓冲区当前存储的经验数量
    def size(self):
        return len(self.buffer)


# 定义 ε-贪心策略，用于根据当前 Q 网络的输出选择动作
def epsilon_greedy_policy(Q_network, state, epsilon, action_dim):
    if random.random() < epsilon:
        # 以 epsilon 概率随机选择动作（探索）
        return random.randint(0, action_dim - 1)
    else:
        # 以 1 - epsilon 概率选择最大 Q 值对应的动作（利用）
        with torch.no_grad():
            return torch.argmax(Q_network(state)).item()


# 初始化环境
env = gym.make('CartPole-v1')

# 获取状态和动作空间的大小
state_dim = env.observation_space.shape[0]  # 状态的维度（输入网络的大小）
action_dim = env.action_space.n  # 动作的数量（输出网络的大小）

# 初始化 Dueling Q 网络和目标网络
q_network = DuelingQNetwork(state_dim, action_dim)  # 当前 Q 网络
target_network = DuelingQNetwork(state_dim, action_dim)  # 目标 Q 网络
target_network.load_state_dict(q_network.state_dict())  # 初始化目标网络，使其与 Q 网络相同

# 设置优化器
optimizer = optim.Adam(q_network.parameters(), lr=0.001)  # 使用 Adam 优化器

# 初始化经验回放缓冲区
replay_buffer = ReplayBuffer(10000)  # 设置缓冲区大小为 10000

# 设置超参数
gamma = 0.99  # 折扣因子，影响未来奖励的权重
epsilon = 1.0  # 初始探索率
epsilon_decay = 0.995  # ε 的衰减率
min_epsilon = 0.01  # 最小探索率
batch_size = 64  # 每次从缓冲区采样的批大小
target_update_freq = 100  # 每隔 100 步更新目标网络
num_episodes = 500  # 总训练 Episode 数量

# 训练循环
for episode in range(num_episodes):
    # 重置环境并获取初始状态
    state, _ = env.reset()
    if isinstance(state, np.ndarray):
        # 检查 state 的长度
        print(f"State shape: {state.shape}")
        state = torch.FloatTensor(state).unsqueeze(0)  # 确保将其转化为 Tensor 并扩展维度
    else:
        print(f"Unexpected state type: {type(state)}")
    #state = torch.FloatTensor(state).unsqueeze(0)  # 将状态转换为 tensor，并扩展维度以符合网络输入格式
    total_reward = 0  # 用于记录每个 episode 的总奖励
    done = False

    while not done:
        # 选择动作
        action = epsilon_greedy_policy(q_network, state, epsilon, action_dim)

        # 执行动作，获取下一个状态、奖励、终止标志等
        next_state, reward, terminated,truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = torch.FloatTensor(next_state).unsqueeze(0)  # 将下一状态转换为 tensor
        total_reward += reward  # 累积奖励

        # 将经验 (state, action, reward, next_state, done) 存入缓冲区
        replay_buffer.push((state, action, reward, next_state, done))

        # 更新当前状态
        state = next_state

        # 如果缓冲区中的经验足够多，则进行训练
        if replay_buffer.size() > batch_size:
            # 从缓冲区中采样 batch_size 大小的经验
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)  # 解包经验
            #for state in states:
               # print(f"State shape: {state.shape}")


            states = torch.cat(states)  # 将所有状态组合成一个 batch
            actions = torch.LongTensor(actions).unsqueeze(1)  # 将动作转换为 tensor
            rewards = torch.FloatTensor(rewards).unsqueeze(1)  # 将奖励转换为 tensor
            next_states = torch.cat(next_states)  # 将下一状态组合成一个 batch
            dones = torch.FloatTensor(dones).unsqueeze(1)  # 将终止标志转换为 tensor

            # 使用目标网络计算下一个状态的最大 Q 值
            next_q_values = target_network(next_states)
            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]  # 获取下一个状态的最大 Q 值
            target_q_values = rewards + gamma * max_next_q_values * (1 - dones)  # 计算目标 Q 值

            # 使用当前 Q 网络计算当前状态的 Q 值
            q_values = q_network(states).gather(1, actions)  # 仅计算当前动作的 Q 值

            # 计算损失并进行优化
            loss = nn.MSELoss()(q_values, target_q_values.detach())  # 使用 MSE 损失函数
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数

        # 每隔一定步数将 Q 网络的参数复制到目标网络
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

    # 衰减 epsilon（减少探索率，逐渐更多利用网络选择的动作）
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}, Total Reward: {total_reward}")  # 打印每个 episode 的总奖励

