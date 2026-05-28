import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# 1. 必须定义和训练时一模一样的网络结构
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


if __name__ == "__main__":
    # 2. 开启人眼看戏模式 (render_mode="human")
    env = gym.make("CartPole-v1", render_mode="human")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    # 3. 初始化网络并加载保存好的权重
    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load("cartpole_dqn_perfect.pth", map_location=device))
    policy_net.eval()  # 切换到测试/评估模式

    print("开始欣赏智能体的表演...（共看 5 回合）")

    for episode in range(5):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_steps = 0

        while True:
            # 4. 纯粹利用 (Exploitation)，没有 Epsilon 随机乱走了
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)

            # 投入环境
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_steps += 1

            if terminated:
                state = None
            else:
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            if done:
                print(f"第 {episode + 1} 回合结束，坚持了 {total_steps} 步！")
                time.sleep(1)  # 每回合结束后稍微停顿一下
                break

    env.close()
    print("播放完毕！")