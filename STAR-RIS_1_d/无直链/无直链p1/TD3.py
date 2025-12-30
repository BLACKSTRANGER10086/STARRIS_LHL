# TD3.py  —— 已替换为 TD3 算法实现

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 600)
        self.fc2 = nn.Linear(600, 400)
        self.fc3 = nn.Linear(400, 400)
        self.out = nn.Linear(400, a_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.out(x))
        return x

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.q1_s = nn.Linear(s_dim, 800)
        self.q1_s_bn = nn.LayerNorm(800)
        self.q1_a = nn.Linear(a_dim, 800)
        self.q1_a_bn = nn.LayerNorm(800)
        self.q1_sa_1 = nn.Linear(800, 600)
        self.q1_sa_2 = nn.Linear(600, 400)
        self.q1_sa_3 = nn.Linear(400, 200)
        self.q1_out = nn.Linear(200, 1)
        # Q2 architecture
        self.q2_s = nn.Linear(s_dim, 800)
        self.q2_s_bn = nn.LayerNorm(800)
        self.q2_a = nn.Linear(a_dim, 800)
        self.q2_a_bn = nn.LayerNorm(800)
        self.q2_sa_1 = nn.Linear(800, 600)
        self.q2_sa_2 = nn.Linear(600, 400)
        self.q2_sa_3 = nn.Linear(400, 200)
        self.q2_out = nn.Linear(200, 1)

    def forward(self, s, a):
        # Q1 forward
        x1 = self.q1_s_bn(self.q1_s(s))
        y1 = self.q1_a_bn(self.q1_a(a))
        net1 = F.relu(x1 + y1)
        net1 = F.relu(self.q1_sa_1(net1))
        net1 = F.relu(self.q1_sa_2(net1))
        net1 = F.relu(self.q1_sa_3(net1))
        q1 = self.q1_out(net1)
        # Q2 forward
        x2 = self.q2_s_bn(self.q2_s(s))
        y2 = self.q2_a_bn(self.q2_a(a))
        net2 = F.relu(x2 + y2)
        net2 = F.relu(self.q2_sa_1(net2))
        net2 = F.relu(self.q2_sa_2(net2))
        net2 = F.relu(self.q2_sa_3(net2))
        q2 = self.q2_out(net2)
        return q1, q2

    def Q1(self, s, a):
        x1 = self.q1_s_bn(self.q1_s(s))
        y1 = self.q1_a_bn(self.q1_a(a))
        net1 = F.relu(x1 + y1)
        net1 = F.relu(self.q1_sa_1(net1))
        net1 = F.relu(self.q1_sa_2(net1))
        net1 = F.relu(self.q1_sa_3(net1))
        return self.q1_out(net1)

class Net(object):
    def __init__(
        self,
        a_dim,
        s_dim,
        pth_path,
        MEMORY_CAPACITY,
        max_action=1.0,
        discount=0.9,
        tau=0.01,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=3
    ):
        self.pth_path = pth_path
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # networks
        self.Actor = Actor(s_dim, a_dim).to(device)
        self.Actor_target = copy.deepcopy(self.Actor).to(device)
        self.actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=3e-4)

        self.Critic = Critic(s_dim, a_dim).to(device)
        self.Critic_target = copy.deepcopy(self.Critic).to(device)
        self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=3e-4)

        # replay buffer
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.loss_fn = nn.MSELoss()
        self.total_it = 0

    def choose_action(self, s):
        self.Actor.eval()
        s = torch.FloatTensor(s).to(device)
        action = self.Actor(s)[0]
        self.Actor.train()
        return action.detach()

    # def store_transition(self, s, a, r, s_):
    #     s = np.squeeze(s)
    #     s_ = np.squeeze(s_)
    #     a = np.squeeze(a)
    #     transition = np.hstack((s, a, [r], s_))
    #     idx = self.pointer % self.MEMORY_CAPACITY
    #     self.memory[idx, :] = transition
    #     self.pointer += 1

    def store_transition(self, s, a, r, s_):
        # 把状态和动作都转成 numpy array 并拍平
        s = np.array(s, dtype=np.float32).reshape(-1)
        s_ = np.array(s_, dtype=np.float32).reshape(-1)
        a = np.array(a, dtype=np.float32).reshape(-1)
        r = float(r)
        transition = np.concatenate([s, a, [r], s_], axis=0)
        idx = self.pointer % self.MEMORY_CAPACITY
        self.memory[idx, :] = transition
        self.pointer += 1

    def learn(self):
        if self.pointer < BATCH_SIZE:
            return
        self.total_it += 1
        # sample batch
        idxs = np.random.randint(0, min(self.pointer, self.MEMORY_CAPACITY), size=BATCH_SIZE)
        bt = self.memory[idxs, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim]).to(device)
        ba = torch.FloatTensor(bt[:, self.s_dim:self.s_dim + self.a_dim]).to(device)
        br = torch.FloatTensor(bt[:, -self.s_dim - 1:-self.s_dim]).to(device)
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:]).to(device)

        # target actions with noise
        with torch.no_grad():
            noise = (torch.randn_like(ba) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a_ = (self.Actor_target(bs_) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.Critic_target(bs_, a_)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = br + self.discount * target_Q

        # current Q estimates
        current_Q1, current_Q2 = self.Critic(bs, ba)
        critic_loss = self.loss_fn(current_Q1, target_Q) + self.loss_fn(current_Q2, target_Q)

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # actor loss
            actor_loss = -self.Critic.Q1(bs, self.Actor(bs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update targets
            for p, tp in zip(self.Critic.parameters(), self.Critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.Actor.parameters(), self.Actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save(self):
        torch.save(self.Actor.state_dict(), self.pth_path + 'actor.pth')
        torch.save(self.Critic.state_dict(), self.pth_path + 'critic.pth')
        print("=========== Models saved ===========")

    def load(self):
        self.Actor.load_state_dict(torch.load(self.pth_path + 'actor.pth'))
        self.Critic.load_state_dict(torch.load(self.pth_path + 'critic.pth'))
        print("=========== Models loaded ===========")
