import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 定义 Actor-Critic 网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 公共层
        self.fc1 = nn.Linear(state_dim, 128)
        # Actor 层
        self.actor = nn.Linear(128, action_dim)
        # Critic 层
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        # Actor 输出动作概率
        action_probs = torch.softmax(self.actor(x), dim=-1)
        # Critic 输出状态值
        state_value = self.critic(x)
        return action_probs, state_value

# 选择动作的函数
def select_action(state, model):
    state = torch.FloatTensor(state).unsqueeze(0)  # 扩展维度
    action_probs, _ = model(state)
    action_dist = torch.distributions.Categorical(action_probs)
    action = action_dist.sample()
    return action.item(), action_dist.log_prob(action)

# 平滑函数
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 初始化 TensorBoard
writer = SummaryWriter(log_dir='runs/actor_critic_training')

# 训练函数
def train_actor_critic(env, model, optimizer, num_episodes=500, gamma=0.99):
    total_rewards = []
    losses = []

    for episode in range(num_episodes):
        state, info = env.reset()
        state = np.array(state)
        log_probs = []
        rewards = []
        state_values = []
        done = False
        total_reward = 0

        while not done:
            action, log_prob = select_action(state, model)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state)
            total_reward += reward

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, state_value = model(state_tensor)
            state_values.append(state_value)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        # 计算折扣奖励
        discounted_rewards = []
        cumulative_reward = 0
        for r in rewards[::-1]:
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        # 将列表转换为张量
        log_probs = torch.stack(log_probs)
        state_values = torch.cat(state_values).squeeze()

        # 计算优势
        advantages = discounted_rewards - state_values.detach()

        # 计算损失
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss

        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_rewards.append(total_reward)
        losses.append(loss.item())

        # 记录到 TensorBoard
        writer.add_scalar('Total Reward', total_reward, episode)
        writer.add_scalar('Loss', loss.item(), episode)

        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    writer.close()

    # 绘制总奖励曲线
    plt.figure()
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Performance over Time')
    plt.show()

    # 绘制平滑的总奖励曲线
    window_size = 10
    smoothed_rewards = moving_average(total_rewards, window_size)
    plt.figure()
    plt.plot(range(window_size - 1, num_episodes), smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Smoothed)')
    plt.title('Smoothed Training Performance over Time')
    plt.show()

    # 绘制损失曲线
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss over Time')
    plt.show()

# 初始化环境和模型
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 开始训练
train_actor_critic(env, model, optimizer, num_episodes=500)
