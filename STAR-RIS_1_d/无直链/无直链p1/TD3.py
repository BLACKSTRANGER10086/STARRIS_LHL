# TD3.py  (FAIR-COMPARE VERSION: same nets & lr as DDPG)

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ====== same hidden size as DDPG ======
hid1 = 1200
hid2 = 800

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, hid1)
        self.bn1 = nn.LayerNorm(hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.bn2 = nn.LayerNorm(hid2)
        self.fc3 = nn.Linear(hid2, hid2)
        self.bn3 = nn.LayerNorm(hid2)
        self.out = nn.Linear(hid2, a_dim)

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = self.bn3(self.fc3(x))
        x = F.relu(x)
        return torch.tanh(self.out(x))

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fcs = nn.Linear(s_dim, hid1)
        self.bn1 = nn.LayerNorm(hid1)
        self.fca = nn.Linear(a_dim, hid1)
        self.bn2 = nn.LayerNorm(hid1)
        self.fc22 = nn.Linear(hid1, hid2)
        self.out = nn.Linear(hid2, 1)

    def forward(self, s, a):
        x = self.bn1(self.fcs(s))
        y = self.bn2(self.fca(a))
        net = F.relu(x + y)
        net = self.fc22(net)
        net = F.relu(net)
        return self.out(net)

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
        policy_freq=2,
        actor_lr=1e-3,
        critic_lr=2e-3
    ):
        self.pth_path = pth_path
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.a_dim = a_dim
        self.s_dim = s_dim

        # Actor
        self.Actor = Actor(s_dim, a_dim).to(device)
        self.Actor_target = copy.deepcopy(self.Actor).to(device)
        self.actor_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=actor_lr)

        # TD3: two critics (same structure as DDPG critic)
        self.Critic1 = Critic(s_dim, a_dim).to(device)
        self.Critic2 = Critic(s_dim, a_dim).to(device)
        self.Critic1_target = copy.deepcopy(self.Critic1).to(device)
        self.Critic2_target = copy.deepcopy(self.Critic2).to(device)
        self.critic_optimizer = torch.optim.Adam(
            list(self.Critic1.parameters()) + list(self.Critic2.parameters()),
            lr=critic_lr
        )

        # replay buffer
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.loss_fn = nn.MSELoss()
        self.total_it = 0

    def choose_action(self, s):
        self.Actor.eval()
        s = torch.as_tensor(s, dtype=torch.float32, device=device)

        if s.dim() == 1:
            s = s.unsqueeze(0)  # [1, s_dim]

        with torch.no_grad():
            a = self.Actor(s)  # [1, a_dim]
        self.Actor.train()

        return a.squeeze(0)  # [a_dim]

    def store_transition(self, s, a, r, s_):
        s = np.array(s, dtype=np.float32).reshape(-1)
        s_ = np.array(s_, dtype=np.float32).reshape(-1)
        a = np.array(a, dtype=np.float32).reshape(-1)
        r = float(r)
        transition = np.concatenate([s, a, [r], s_], axis=0)
        idx = self.pointer % self.MEMORY_CAPACITY
        self.memory[idx, :] = transition
        self.pointer += 1

    def learn(self):
        max_mem = min(self.pointer, self.MEMORY_CAPACITY)
        if max_mem < BATCH_SIZE:
            return

        self.total_it += 1
        idxs = np.random.randint(0, max_mem, size=BATCH_SIZE)
        bt = self.memory[idxs, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim]).to(device)
        ba = torch.FloatTensor(bt[:, self.s_dim:self.s_dim + self.a_dim]).to(device)
        br = torch.FloatTensor(bt[:, -self.s_dim - 1:-self.s_dim]).to(device)
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:]).to(device)

        # target actions with smoothing noise (TD3 feature)
        with torch.no_grad():
            noise = (torch.randn_like(ba) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a_ = (self.Actor_target(bs_) + noise).clamp(-self.max_action, self.max_action)

            target_Q1 = self.Critic1_target(bs_, a_)
            target_Q2 = self.Critic2_target(bs_, a_)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = br + self.discount * target_Q

        current_Q1 = self.Critic1(bs, ba)
        current_Q2 = self.Critic2(bs, ba)
        critic_loss = self.loss_fn(current_Q1, target_Q) + self.loss_fn(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # delayed policy updates (TD3 feature)
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.Critic1(bs, self.Actor(bs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update targets
            for p, tp in zip(self.Actor.parameters(), self.Actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.Critic1.parameters(), self.Critic1_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.Critic2.parameters(), self.Critic2_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def save(self):
        torch.save(self.Actor.state_dict(), self.pth_path + 'actor.pth')
        torch.save(self.Critic1.state_dict(), self.pth_path + 'critic1.pth')
        torch.save(self.Critic2.state_dict(), self.pth_path + 'critic2.pth')
        print("=========== Models saved ===========")

    def load(self):
        self.Actor.load_state_dict(torch.load(self.pth_path + 'actor.pth'))
        self.Critic1.load_state_dict(torch.load(self.pth_path + 'critic1.pth'))
        self.Critic2.load_state_dict(torch.load(self.pth_path + 'critic2.pth'))
        print("=========== Models loaded ===========")
