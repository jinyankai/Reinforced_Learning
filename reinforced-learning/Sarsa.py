import gym
import numpy as np
import random
import matplotlib.pyplot as plt

# 创建环境
env = gym.make('CliffWalking-v0')

# 获取状态和动作空间的大小
state_dim = env.observation_space.n
action_dim = env.action_space.n

# 超参数
alpha = 0.5      # 学习率
gamma = 0.9      # 折扣因子
epsilon = 0.1    # ε-贪心策略的探索率
num_episodes = 500  # 训练的 Episode 数

# 定义 ε-贪心策略
def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        # 随机选择动作（探索）
        action = env.action_space.sample()
    else:
        # 选择具有最大 Q 值的动作（利用）
        action = np.argmax(Q[state, :])
    return action

# 初始化 Q 值表
Q = np.zeros((state_dim, action_dim))

# 用于记录每个 Episode 的总奖励
rewards_per_episode = []

for episode in range(num_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # 提取元组中的第一个元素

    print(f"Initial state: {state}, Type: {type(state)}")  # 调试信息，检查状态

    total_reward = 0
    done = False

    # 选择初始动作
    action = epsilon_greedy_policy(Q, state, epsilon)

    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 选择下一个动作（基于当前策略）
        next_action = epsilon_greedy_policy(Q, next_state, epsilon)

        # 更新 Q 值
        Q[state, action] += alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        state = next_state
        action = next_action
        total_reward += reward

    rewards_per_episode.append(total_reward)

    # 可选：逐渐减少 ε，提高策略的贪心程度
    # epsilon = max(0.01, epsilon * 0.995)

    # 打印训练进度
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")

# 绘制每个 Episode 的总奖励
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('SARSA on CliffWalking-v0')
plt.show()

# 测试训练好的策略
state = env.reset()
done = False
env.render()

while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, terminated, truncated, _ = env.step(action)
    env.render()
