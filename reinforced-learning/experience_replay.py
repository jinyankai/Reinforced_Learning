import random
from collections import deque

import gym


# 定义经验回放缓冲区类
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 使用 deque 存储经验，最大长度为 capacity

    # 将经验 (state, action, reward, next_state, done) 加入缓冲区
    def push(self, experience):
        self.buffer.append(experience)

    # 从缓冲区中随机采样 batch_size 个经验
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    # 返回缓冲区当前的大小
    def size(self):
        return len(self.buffer)

# 使用示例
buffer = ReplayBuffer(capacity=10000)
# env = gym.make('CartPole-v1')
# 将经验 (state, action, reward, next_state, done) 加入缓冲区
state, action, reward, next_state, done = 0, 1, 1.0, 2, False
# state, reward,terminated,truncated,_ = env.step(action)
# done = terminated or truncated
buffer.push((state, action, reward, next_state, done))

# 从缓冲区中采样一个 batch
batch_size = 32
if buffer.size() > batch_size:
    batch = buffer.sample(batch_size)
    print(batch)
