import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BATCH_SIZE = 64
TAU = 0.1      # soft replacement
GAMMA = 0.9     # reward_plot discount
c_lr = 0.002
a_theta_lr = 0.001
a_w_lr = 0.001

# hid1 = 1100
# hid2 = 800

hid1 = 1200
hid2 = 800

device = torch.device("cuda:0")
# device = torch.device("cpu")


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, hid1)
        self.bn1 = nn.LayerNorm(hid1)  # nn.BatchNorm1d
        self.fc2 = nn.Linear(hid1, hid2)

        self.res1 = nn.Linear(hid2, hid2)
        self.res2 = nn.Linear(hid2, hid2)

        self.bn2 = nn.LayerNorm(hid2)  # nn.BatchNorm1d
        self.fc3 = nn.Linear(hid2, hid2)
        # self.fc3.weight.data.normal_(0, 0.1)
        self.bn3 = nn.LayerNorm(hid2)  # nn.BatchNorm1d
        self.out = nn.Linear(hid2, a_dim)

    def forward(self, x):
        ##
        x = self.fc1(x)
        x = self.bn1(x)
        # x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)

        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        # return np.pi * x
        return x


class Critic(nn.Module):
    def __init__(self,s_dim,a_dim):
        super(Critic,self).__init__()
        self.fcs = nn.Linear(s_dim, hid1)
        self.bn1 = nn.LayerNorm(hid1)
        self.fca = nn.Linear(a_dim, hid1)
        self.bn2 = nn.LayerNorm(hid1)

        self.res1 = nn.Linear(hid1, hid1)
        self.res2 = nn.Linear(hid1, hid1)

        self.fc22 = nn.Linear(hid1, hid2)
        self.bn22 = nn.LayerNorm(hid2)
        self.fc33 = nn.Linear(hid2, hid2)
        self.bn33 = nn.LayerNorm(hid2)

        self.res3 = nn.Linear(hid2, hid2)

        self.out = nn.Linear(hid2, 1)

    def forward(self,s,a):
        x = self.fcs(s)
        x = self.bn1(x)
        y = self.fca(a)
        y = self.bn2(y)
        net = F.relu(x+y)
        net = self.fc22(net)

        net = F.relu(net)
        Q_value = self.out(net)
        return Q_value


class Net(object):
    def __init__(self, a_dim, s_dim, pth_path, MEMORY_CAPACITY):
        self.pth_path = pth_path
        self.Actor_eval = Actor(s_dim, a_dim).to(device)
        self.Actor_target = Actor(s_dim, a_dim).to(device)
        self.Critic_eval = Critic(s_dim, a_dim).to(device)
        self.Critic_target = Critic(s_dim, a_dim).to(device)

        # self add
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.memory = np.zeros((self.MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.pointer_dqn = 0
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=c_lr)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=a_theta_lr)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        self.Actor_eval.eval()
        s = torch.FloatTensor(s).to(device)
        # w_amp = w_amp / torch.norm(w_amp[0], p=2, dim=1)
        return self.Actor_eval(s)[0].detach()

    def store_transition(self, s, a, r, s_):
        s = np.squeeze(s)
        s_ = np.squeeze(s_)
        a = np.squeeze(a)
        r = np.squeeze(r)
        # print(type(s),type(s_),type(a),type(r))
        # print(s.shape, s_.shape, a.shape, r.shape)
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def learn_theta(self):
        for param, target_param in zip(self.Critic_eval.parameters(), self.Critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.Actor_eval.parameters(), self.Actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        indices = np.random.choice(self.MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]  # ;print(bt.shape)  #
        bs = torch.FloatTensor(bt[:, :self.s_dim]).to(device)  # ;print(bs.shape) #
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim]).to(device)  # ;print(ba.shape) #
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim]).to(device)  # ;print(br.shape)#
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:]).to(device)  # ;print(bs_.shape) #

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs, a)
        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()  # 更新actor参数

        a_ = self.Actor_target(bs_)
        q_ = self.Critic_target(bs_, a_)
        q_target = br + GAMMA * q_  # print(q_target)
        q_v = self.Critic_eval(bs, ba)  # print(q_v)
        td_error = self.loss_td(q_target, q_v)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()  # 更新critic参数

    def save(self):
        torch.save(self.Actor_eval.state_dict(), self.pth_path + 'actor.pth')
        torch.save(self.Critic_eval.state_dict(), self.pth_path + 'critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.Actor_eval.load_state_dict(torch.load(self.pth_path + 'actor.pth'))
        self.Critic_eval.load_state_dict(torch.load(self.pth_path + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
