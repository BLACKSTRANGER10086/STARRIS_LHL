import math
import gymnasium as gym
import numpy as np
from scipy.special import erfinv

class RISEnvironment(gym.Env):
    def __init__(self):
        super(RISEnvironment, self).__init__()
        # 参数
        self.K0 = 5
        self.K = [5, 5]
        # self.K0 = 10
        # self.K = [10, 10]
        self.M = 8  # BS天线数
        self.N = 128  # RIS单元数
        # self.M = 8  # BS天线数
        # self.N = 64  # RIS单元数
        self.Rho_ = 50
        self.Rho = 10 ** (self.Rho_ / 10)

        # BS-IRS 信道角度设置
        self.theta_AOA0 = np.pi / 9
        self.theta_AOD0 = np.pi / 3

        # STAR-RIS ES模式下定义两个用户
        # 透射侧 反射侧
        # self.theta_AOD = [np.pi / 6, np.pi / 4]
        self.theta_AODt = [np.pi / 180 * 120, np.pi / 180 * 100]
        self.theta_AODr = [np.pi / 6, np.pi / 4]
        self.U1 = len(self.theta_AODt)
        self.U2 = len(self.theta_AODr)  # 总用户数
        self.U = self.U1 + self.U2  # 总用户数

        self.c = 3e8
        self.fre = 1e9
        self.lamda = self.c / self.fre
        self.d = self.lamda / 2

        # 动作维度
        self.act_dim = 3 * self.N  # 包括透射和反射相位 以及透射和反射系数
        # 状态维度
        # elf.state_dim = 2 * self.N * self.M + self.U * self.N * 2 + self.M * 2 + self.N * 4 + 2 + self.U
        self.state_dim = 2 * self.N * self.M + 2 * self.U1 * self.N + 2 * self.U2 * self.N + 2 * self.M + 4 * self.N + 2 + self.U

        self.steps = 200

        # 路径损耗参数
        self.path_loss_ref = 0.01
        self.path_loss_e_br = 2.2
        self.path_loss_e_eu_t = 2.8
        self.path_loss_e_eu_r = 2.8

        # 距离设置
        # self.d0 = 40 / 10  # BS到IRS 距离
        self.d0 = 40 / 10  # BS到IRS 距离
        # 反射和透射各两个用户
        # self.dt = [3,5]
        # self.dr = [4,6]
        self.dt = [2,3]
        self.dr = [4,5  ]

        # 计算透射反射路径损耗
        self.pHi0 = self.calculate_path_loss(self.path_loss_ref, self.d0, self.path_loss_e_br)
        self.pHi_t = []
        self.pHi_r = []
        for d_val_t in self.dt:
            self.pHi_t.append(self.calculate_path_loss(self.path_loss_ref, d_val_t, self.path_loss_e_eu_t))
        for d_val_r in self.dr:
            self.pHi_r.append(self.calculate_path_loss(self.path_loss_ref, d_val_r, self.path_loss_e_eu_r))

        # BS-STAR-IRS信道
        self.g_los_ = self.aN_Theta(self.N, self.theta_AOA0)
        self.g_los = self.aN_Theta(self.N, self.theta_AOA0).conjugate().T @ self.aN_Theta(self.M, self.theta_AOD0)

        # STAR-IRS透射信道
        self.h_los_t = []
        for theta in self.theta_AODt:
            self.h_los_t.append(self.aN_Theta(self.N, theta))

        # STAR-IRS反射信道
        self.h_los_r = []
        for theta in self.theta_AODr:
            self.h_los_r.append(self.aN_Theta(self.N, theta))

        # 各用户的最优相位
        self.ris_angle_std_r = []
        self.ris_angle_std_t = []
        for hr in self.h_los_r:
            self.ris_angle_std_r.append(-(np.angle(hr) - np.angle(self.g_los_)))
        for ht in self.h_los_t:
            self.ris_angle_std_t.append(-(np.angle(ht) - np.angle(self.g_los_)))

        # 波束赋形矢量设计
        self.f_up = self.h_los_r[0] @ np.diag(np.exp(1j * self.ris_angle_std_r[0]).flatten()) @ self.g_los
        self.f_down = np.linalg.norm(self.f_up)
        self.f = self.f_up / self.f_down

        # 构造参数列表
        self.user_params_t = [
            {
                "theta_AODt": self.theta_AODt[i],
                "pHit": self.pHi_t[i],
                "distance_t": self.dt[i],
                "K": self.K[i],
                "f": self.f,
                "h_los_t": self.h_los_t[i],
                "ris_angle_std": self.ris_angle_std_t[i]
            }
            for i in range(self.U1)
        ]

        self.user_params_r = [
            {
                "theta_AODr": self.theta_AODr[i],
                "pHir": self.pHi_r[i],
                "distance_r": self.dr[i],
                "K": self.K[i],
                "f": self.f,
                "h_los_r": self.h_los_r[i],
                "ris_angle_std": self.ris_angle_std_r[i]
            }
            for i in range(self.U2)
        ]

    # 环境重置
    def reset(self):
        state = []
        g_los_ = self.g_los_.reshape(-1)
        g_los = self.g_los.reshape(-1)
        state.append(np.hstack((np.real(g_los), np.imag(g_los))))

        # STAR_IRS-透射用户信道信息
        for user in self.user_params_t:
            h_los_t = user["h_los_t"].reshape(-1)
            state.append(np.hstack((np.real(h_los_t), np.imag(h_los_t))))

        # STAR_IRS-反射用户信道信息
        for user in self.user_params_r:
            h_los_r = user["h_los_r"].reshape(-1)
            state.append(np.hstack((np.real(h_los_r), np.imag(h_los_r))))

        theta_AOA0 = np.array(self.theta_AOA0).reshape(-1)
        theta_AOD0 = np.array(self.theta_AOD0).reshape(-1)

        ris_angle_t = (-np.angle(g_los_) + np.angle(self.user_params_t[0]["h_los_t"])).reshape(-1)
        ris_amp_t = self.Beta_Min(ris_angle_t).reshape(-1)

        ris_angle_r = (-np.angle(g_los_) + np.angle(self.user_params_r[0]["h_los_r"])).reshape(-1)
        ris_amp_r = self.Beta_Min(ris_angle_r).reshape(-1)

        f = self.f.reshape(-1)

        # 构造状态
        state.append(np.hstack((np.real(f), np.imag(f), ris_angle_t, ris_amp_t, ris_angle_r, ris_amp_r, theta_AOA0, theta_AOD0)))

        for user in self.user_params_t:
            theta_AODt = np.array(user["theta_AODt"]).reshape(-1)
            state.append(theta_AODt)

        for user in self.user_params_r:
            theta_AODr = np.array(user["theta_AODr"]).reshape(-1)
            state.append(theta_AODr)

        state = np.concatenate(state)
        # print(state.shape)
        return np.array([state]).astype(np.float32)

    # 获得环境
    def get_state(self, ris_angle_t, ris_angle_r, ris_amp_t, ris_amp_r, f):
        state = []
        g_los = self.g_los.reshape(-1)
        state.append(np.hstack((np.real(g_los), np.imag(g_los))))

        # STAR_IRS-透射用户信道信息
        for user in self.user_params_t:
            h_los_t = user["h_los_t"].reshape(-1)
            state.append(np.hstack((np.real(h_los_t), np.imag(h_los_t))))

        # STAR_IRS-反射用户信道信息
        for user in self.user_params_r:
            h_los_r = user["h_los_r"].reshape(-1)
            state.append(np.hstack((np.real(h_los_r), np.imag(h_los_r))))

        theta_AOA0 = np.array(self.theta_AOA0).reshape(-1)
        theta_AOD0 = np.array(self.theta_AOD0).reshape(-1)

        ris_angle_t = ris_angle_t.reshape(-1)
        ris_amp_t = ris_amp_t.reshape(-1)
        ris_angle_r = ris_angle_r.reshape(-1)
        ris_amp_r = ris_amp_r.reshape(-1)

        f = f.reshape(-1)

        state.append(np.hstack((np.real(f), np.imag(f), ris_angle_t, ris_amp_t, ris_angle_r, ris_amp_r, theta_AOA0, theta_AOD0)))

        for user in self.user_params_t:
            theta_AODt = np.array(user["theta_AODt"]).reshape(-1)
            state.append(theta_AODt)

        for user in self.user_params_r:
            theta_AODr = np.array(user["theta_AODr"]).reshape(-1)
            state.append(theta_AODr)

        # 拼接成一个一维数组
        state = np.concatenate(state)
        # print(state.shape)
        return np.array([state]).astype(np.float32)

    # 环境更新
    def step(self, action):

        action = action.reshape(-1)
        ris_angle_t = action[:self.N] * np.pi
        ris_angle_r = action[self.N:self.N*2] * np.pi

        # 透射和反射系数
        beta_r = (action[2*self.N:3*self.N]+1)/2
        beta_t = np.sqrt(1-beta_r**2)

        # 理想相移模型 全1
        ris_amp_t = self.Beta_Min(ris_angle_t)
        ris_amp_r = self.Beta_Min(ris_angle_r)

        # 波束赋形
        f = self.f.reshape(1, -1)

        # 计算透射用户奖励
        rewards_t = []
        for idx, user in enumerate(self.user_params_t):
            reward_t = self.CalStaSNR_TongJi(
                h1_los=self.g_los,
                h2_los=user["h_los_t"],
                f=f,
                beta=beta_t,
                ris_amp=ris_amp_t,
                ris_angle=ris_angle_t,
                k1=self.K0,
                k2=user["K"],
                pHi0=self.pHi0,
                pHi1=user["pHit"],
            )
            rewards_t.append(reward_t)
        rewards_t = np.array(rewards_t).reshape(-1)

        # 计算反射用户奖励
        rewards_r = []
        for idx, user in enumerate(self.user_params_r):
            reward_r = self.CalStaSNR_TongJi(
                h1_los=self.g_los,
                h2_los=user["h_los_r"],
                f=f,
                beta=beta_r,
                ris_amp=ris_amp_r,
                ris_angle=ris_angle_r,
                k1=self.K0,
                k2=user["K"],
                pHi0=self.pHi0,
                pHi1=user["pHir"],
            )
            rewards_r.append(reward_r)
        rewards_r = np.array(rewards_r).reshape(-1)

        # max_min 信噪比
        all_rewards = np.concatenate([rewards_t, rewards_r])

        # 最终奖励：所有用户、所有模式下的最小速率
        reward = np.min(all_rewards)
        # print(reward)

        # 计算反射用户标准奖励
        reward_stds_r = []
        for idx, user in enumerate(self.user_params_r):
            reward_std = self.CalStaSNR_TongJi(
                h1_los=self.g_los,
                h2_los=user["h_los_r"],
                f=f,
                beta=beta_r,
                ris_amp=ris_amp_r,
                ris_angle=user["ris_angle_std"],
                k1=self.K0,
                k2=user["K"],
                pHi0=self.pHi0,
                pHi1=user["pHir"],
            )
            reward_stds_r.append(reward_std)

        # 计算透射用户标准奖励
        reward_stds_t = []
        for idx, user in enumerate(self.user_params_t):
            reward_std = self.CalStaSNR_TongJi(
                h1_los=self.g_los,
                h2_los=user["h_los_t"],
                f=f,
                beta=beta_t,
                ris_amp=ris_amp_t,
                ris_angle=user["ris_angle_std"],
                k1=self.K0,
                k2=user["K"],
                pHi0=self.pHi0,
                pHi1=user["pHit"],
            )
            reward_stds_t.append(reward_std)

        # 构造下一个状态
        next_state = self.get_state(ris_angle_t=ris_angle_t, ris_angle_r=ris_angle_r, ris_amp_t=ris_amp_t, ris_amp_r=ris_amp_r, f=f)
        next_state = np.array(next_state).astype(np.float32)
        # print(next_state.shape)
        return next_state, reward, rewards_t, rewards_r, ris_angle_t, ris_amp_t, ris_angle_r, ris_amp_r, f, beta_r, beta_t, reward_stds_t[0],  reward_stds_r[0], reward_stds_t[self.U1-1],  reward_stds_r[self.U2-1]

    # 计算奖励
    def CalStaSNR_TongJi(self, h1_los, h2_los, f, beta, ris_amp, ris_angle, k1, k2, pHi0, pHi1):
        kc = (k1 + 1) * (k2 + 1)
        ris_amp_angle = np.diag((ris_amp * beta * np.exp(1j * ris_angle)).flatten())
        A1 = pHi0 * pHi1 * np.sqrt((k1 * k2) / kc) * np.abs((h2_los @ ris_amp_angle @ h1_los @ f.conjugate().T))
        ac = np.abs(self.aN_Theta(self.M, self.theta_AOD0) @ f.conjugate().T)
        EH2 = np.abs(A1) ** 2 + (pHi0 * pHi1) ** 2 * ((ac ** 2 * k1 + k2 + 1) * np.sum((ris_amp*beta) ** 2)) / kc
        gamma = self.Rho * EH2
        R = np.log2(1 + gamma) - self.inv_qfunc(1e-5) / (np.sqrt(200) * np.log10(2))
        return R

    # 计算阵列相应
    def aN_Theta(self, angel_num, angel):
        aN_Theta = []
        for i in range(angel_num):
            aN_Theta.append(np.exp(1j * np.pi * np.sin(angel) * i))
        return np.array(aN_Theta).reshape(1, -1)

    # 理想相移模型下幅度函数 全1
    def Beta_Min(self, ris_angle):
        return np.ones_like(ris_angle)

    # 波束赋形矢量
    def Beamforming(self, g_los, ris_amp, ris_angle, h_los):
        f_h = h_los @ (np.diag((ris_amp * np.exp(1j * ris_angle)).flatten())) @ g_los
        f_h_abs = np.linalg.norm(f_h)
        f = f_h / f_h_abs
        return f

    # 计算路径损耗
    def calculate_path_loss(self, path_loss_ref, d, loss_exponent):
        return np.sqrt(path_loss_ref * d ** (-loss_exponent))

    # 计算Q逆函数
    def inv_qfunc(self, y):
        return np.sqrt(2) * erfinv(1 - 2 * y)