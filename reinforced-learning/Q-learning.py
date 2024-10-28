import gym
import matplotlib as plt
import numpy as np
import random
env = gym.make('FrozenLake-v1')
state_dim = env.observation_space.n
action_dim = env.action_space.n
# 超参数
alpha = 0.1  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 0.1  # ε-贪心策略的探索率
num_episodes = 1000  # 训练的 Episode 数
max_steps_per_episode = 100  # 每个 episode 最大步数

Q = np.zeros((state_dim,action_dim))
# epsilon_greedy
def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        # 随机选择动作（探索）
        return env.action_space.sample()
    else:
        # 选择具有最大 Q 值的动作（利用）
        return np.argmax(Q[state, :])

rewards_per_episode = []

for episode in range(num_episodes):
    state = env.reset()
    if isinstance(state,tuple):
        state = state[0]
    total_reward = 0
    done = False

    for step in range(max_steps_per_episode):
        # 根据 ε-贪心策略选择动作
        action = epsilon_greedy_policy(Q, state, epsilon)

        # 执行动作，获取新的状态和奖励
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 更新 Q 值
        Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        # 将状态更新为下一个状态
        state = next_state
        total_reward += reward

        if done:
            break

    rewards_per_episode.append(total_reward)

    # 可选：逐渐减少 ε，提高策略的利用
    # epsilon = max(0.01, epsilon * 0.995)

    # 打印训练结果
    print(f"Training completed over {num_episodes} episodes.")
