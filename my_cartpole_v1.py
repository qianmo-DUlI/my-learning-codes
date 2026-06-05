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


#使用human可以看到训练动画，关掉加快速度
#env = gym.make("CartPole-v1",render_mode ="human")
env = gym.make("CartPole-v1")

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

#GPU设置
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


#在测试中固定种子
seed = 42
random.seed(seed)
env.action_space.seed(seed=seed)
env.observation_space.seed(seed=seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


#下面实现replay memory
#用namedtuple防止使用0，1，2，3这样的下标自己分不清
Transition = namedtuple("Transition",("state","action","next_state","reward"))
class ReplayMemory:
    def __init__(self, capacity):
        #用的是双端队列，但只使用他的栈的功能
        self.memory = deque([],maxlen=capacity)
        self.capacity = capacity

        #将新的数据（这个数据是与环境交互时产生的）入栈
    def push(self,*args):
        self.memory.append(Transition(*args))

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)


#神经网络构建
class DQN(nn.Module):
    def __init__(self,n_observations,n_actions):
        super(DQN,self).__init__()
        self.layer1 =nn.Linear(n_observations,128)
        self.layer2 =nn.Linear(128,128)
        self.layer3 =nn.Linear(128,n_actions)

    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        return self.layer3(x)

if __name__=="__main__":
    BATCH_SIZE=128          #每次训练所需要从replay memory里面采集的数量
    GAMMA=0.99              #γ折扣因子
    EPSILON_START =0.9      #初始的探索率
    EPSILON_END =0.01       #最终的探索率
    EPSILON_DECAY=2500      #探索率epsilon的衰减速度
    LR =3e-4                #学习率
    TARGET_UPDATE=200       #每C步更新目标网络

    n_actions =env.action_space.n   #动作数这里是2
    state,info =env.reset()         #重置环境到初始状态，返回初始状态和额外信息
    n_observations=len(state)       #获取状态维度这里是
    # 初始化双Q网络
    policy_net=DQN(n_observations,n_actions).to(device) # 策略网络：实时更新，负责选动作
    target_net=DQN(n_observations,n_actions).to(device) # 目标网络：延迟更新，负责计算目标Q值
    target_net.load_state_dict(policy_net.state_dict()) # 初始参数完全相同
    # 初始化优化器和经验回放池
    optimizer =optim.AdamW(policy_net.parameters(),lr=LR,amsgrad=True)
    memory =ReplayMemory(10000)


    # 全局步数计数器，记录总交互步数
    steps_done =0

    #ε-greedy
    def select_action(state):
        global steps_done
        sample = random.random()
        #计算当前探索率
        eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1. * steps_done / EPSILON_DECAY)
        steps_done += 1
        #以1-ε概率选择最优
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
         #ε概率随机选一个
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


    # 记录每个回合的坚持步数
    episode_durations = []


    def plot_durations(show_result=False):
        plt.figure(1)
        #---------
        plt.clf()
        #-------
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        #绘制100回合的平均
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        #随机从reply  memory选取BATCH_SIZE大的样本来训练主网络
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))      #[(s1,a1,s1',r1), (s2,a2,s2',r2)] → ((s1,s2), (a1,a2), (s1',s2'), (r1,r2))
        #标记下一个状态是否是终止状态，也就是生成一个布尔张量如果下一个状态是None，则是该位置是True
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        #过滤掉None（终止状态没有下一个状态），把剩下的所有非终止状态拼接成一个可以直接输入神经网络的Batch张量。
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        #计算y_T
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # 计算当前的Q值     实际执行动作的价值
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        #Huber损失
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # 反向传播更新主网络参数w
        optimizer.zero_grad()  # 清空上一步的梯度
        loss.backward()  # 计算梯度
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()  # 更新参数


    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 100

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()
            #每TARGET_UPDATE步更新一次目标网络
            if steps_done % TARGET_UPDATE ==0:
               target_net.load_state_dict(policy_net.state_dict())
            if done:
                episode_durations.append(t + 1)
                #plot_durations()
                # 添加控制台日志
                if (i_episode + 1) % 10 == 0:
                    avg_duration = sum(episode_durations[-10:]) / 10
                    print(f"Episode [{i_episode + 1}/{num_episodes}], 最近10回合平均步数: {avg_duration:.1f}")
                break

    print('Complete')
    plot_durations(show_result=True)
    #保存训练好的模型权重
    #torch.save(policy_net.state_dict(), "cartpole_dqn_perfect.pth")
    #print("模型权重已成功保存为: cartpole_dqn_perfect.pth")
    plt.ioff()
    plt.show()






