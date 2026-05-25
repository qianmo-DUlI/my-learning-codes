import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# env = gym.make("CartPole-v1")
#
# # 设置 matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display
#
# plt.ion()
#
# # 如果要使用 GPU
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else
#     "mps" if torch.backends.mps.is_available() else
#     "cpu"
# )


# 创建训练环境——带杆子的推车，需要保持平衡
env = gym.make("CartPole-v1", render_mode="human")
# 重置环境以开始新回合
observation, info = env.reset()
# observation：智能体能“看到”的内容——小车位置、速度、杆角度等
# info：额外调试信息（基本学习通常不需要）
print(f"Starting observation: {observation}")
# 示例输出: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
episode_over = False
total_reward = 0
while not episode_over:
    # 选择动作：0 = 向左推车，1 = 向右推车
    action = env.action_space.sample()  # 目前随机选择动作——真实智能体会更智能！
    # 执行动作并观察结果
    observation, reward, terminated, truncated, info = env.step(action)
    # reward：杆保持直立时每一步 +1
    # terminated：杆倾倒太远时为 True（智能体失败）
    # truncated：达到时间限制（500 步）时为 True
    total_reward += reward
    episode_over = terminated or truncated
print(f"Episode finished! Total reward: {total_reward}")
env.close()

