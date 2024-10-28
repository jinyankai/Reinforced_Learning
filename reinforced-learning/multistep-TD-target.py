import gym
import numpy as np
import random
from collections import deque

# 创建环境
env = gym.make('CartPole-v1')

# 获取状态和动作空间的大小
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 定义每个状态变量的离散化区间
cart_position_bins = np.linspace(-2.4, 2.4, 10)  # 小车位置的区间划分
cart_velocity_bins = np.linspace(-3.0, 3.0, 10)  # 小车速度的区间划分
pole_angle_bins = np.linspace(-0.209, 0.209, 10)  # 杆的角度的区间划分
pole_velocity_bins = np.linspace(-2.0, 2.0, 10)  # 杆的速度的区间划分

bins = [cart_position_bins, cart_velocity_bins, pole_angle_bins, pole_velocity_bins]

# 超参数
alpha = 0.1  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 0.1  # ε-贪心策略的探索率
n = 3  # n-step TD 的步数
num_episodes = 500  # 总训练 Episode 数

# 初始化 Q 值表，假设每个状态的离散化区间为 10 个
Q = np.zeros((10, 10, 10, 10, action_dim))  # 4 个离散状态维度 + 动作维度


# 定义状态离散化函数
def discretize_state(state, bins):
    """将连续状态变量离散化为离散值"""
    discrete_state = tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))
    return discrete_state


# 定义 ε-贪心策略
def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# 记录每个 Episode 的总奖励
rewards_per_episode = []

for episode in range(num_episodes):
    state = env.reset()

    state = discretize_state(state, bins)  # 将连续状态离散化
    total_reward = 0
    done = False
    t = 0

    # 存储 n 步的经验 (state, action, reward)
    n_step_buffer = deque(maxlen=n)

    while not done:
        action = epsilon_greedy_policy(Q, state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state, bins)  # 将连续状态离散化

        # 将经验存入 n-step 缓冲区
        n_step_buffer.append((state, action, reward))

        # 如果缓冲区满了，开始更新
        if len(n_step_buffer) == n:
            G = 0
            for i in range(n):
                G += (gamma ** i) * n_step_buffer[i][2]  # 累积 n 步奖励
            G += (gamma ** n) * np.max(Q[next_state])  # 加上未来的 Q 值估计

            # 更新 Q 值
            state_update, action_update, _ = n_step_buffer[0]  # 第一个 state 和 action
            Q[state_update][action_update] += alpha * (G - Q[state_update][action_update])

        state = next_state
        total_reward += reward
        t += 1

    rewards_per_episode.append(total_reward)

    # 打印训练进度
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# 训练结束
print("Training completed.")
