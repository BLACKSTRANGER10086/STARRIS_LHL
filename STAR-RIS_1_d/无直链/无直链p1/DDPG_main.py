import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
from STARRIS_ENV import RISEnvironment
from DDPG import Net
import time
import torch
from scipy.io import savemat

print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.is_available())

max_episode = 400
MEMORY_CAPACITY = 10000
var = 1

env = RISEnvironment()
steps = env.steps
ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk05k5/m4n64/', MEMORY_CAPACITY=MEMORY_CAPACITY)
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk05k5/m4n96/', MEMORY_CAPACITY=MEMORY_CAPACITY)
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk05k5/m4n128/', MEMORY_CAPACITY=MEMORY_CAPACITY)
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk05k5/m8n64/', MEMORY_CAPACITY=MEMORY_CAPACITY)
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk05k5/m8n96/', MEMORY_CAPACITY=MEMORY_CAPACITY)
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk05k5/m8n128/', MEMORY_CAPACITY=MEMORY_CAPACITY)
#
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk010k10/m4n64/', MEMORY_CAPACITY=MEMORY_CAPACITY)
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk010k10/m4n96/', MEMORY_CAPACITY=MEMORY_CAPACITY)
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk010k10/m4n128/', MEMORY_CAPACITY=MEMORY_CAPACITY)
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk010k10/m8n64/', MEMORY_CAPACITY=MEMORY_CAPACITY)
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk010k10/m8n96/', MEMORY_CAPACITY=MEMORY_CAPACITY)
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk010k10/m8n128/', MEMORY_CAPACITY=MEMORY_CAPACITY)

# 记录奖励
rewards_plot = []
reward1_plot = []
reward2_plot = []
reward3_plot = []
reward4_plot = []
reward5_plot = []
reward6_plot = []
reward7_plot = []
reward8_plot = []

time_start_1 = time.time()

# 如果需要加载预训练模型，则取消注释
# ddpg.load()

for i in range(max_episode):
    reward_total = 0
    rt_total1 = 0
    rr_total1 = 0
    rt_total2 = 0
    rr_total2 = 0
    rt_std1_total = 0
    rr_std1_total = 0
    rt_std2_total = 0
    rr_std2_total = 0
    s = env.reset()
    # print(s)
    for j in range(steps):
        # 选择动作
        a = ddpg.choose_action(s)
        a = a.cpu().numpy()
        a = np.clip(np.random.normal(a, var), -1, 1)
        s_, r, rt, rr, ris_angle_T, ris_amp_T, ris_angle_R, ris_amp_R, f, New_beta_R, New_beta_T, rt_std1, rr_std1, rt_std2, rr_std2 = env.step(a)
        # print(rt)
        # print(rr)
        # print("\nReward (r):", r)
        # print("Reward for transmission user (rt):", rt)
        # print("Reward for reflection user (rr):", rr)
        ddpg.store_transition(s, a, r, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= 0.9999
            ddpg.learn_theta()
        s = s_
        reward_total += r  # 累加总奖励
        rt_total1 += rt[0]
        rr_total1 += rr[0]
        rt_total2 += rt[1]
        rr_total2 += rr[1]
        rt_std1_total += rt_std1
        rr_std1_total += rr_std1
        rt_std2_total += rt_std2
        rr_std2_total += rr_std2
    # print(rt_total)
    # print(rr_total)
    print('Episode:', i,
        '| \033[1;35mr: %f\033[0m' % (reward_total / steps).item(),
        '| \033[1;36mrt1: %f\033[0m' % (rt_total1 / steps).item(),
        '| \033[1;36mrr1: %f\033[0m' % (rr_total1 / steps).item(),
          '| \033[1;36mrt2: %f\033[0m' % (rt_total2 / steps).item(),
          '| \033[1;36mrr2: %f\033[0m' % (rr_total2 / steps).item(),
          '| \033[1;35mrt_std1: %f\033[0m' % (rt_std1_total / steps).item(),
          '| \033[1;35mrr_std1: %f\033[0m' % (rr_std1_total / steps).item(),
          '| \033[1;35mrt_std2: %f\033[0m' % (rt_std2_total / steps).item(),
          '| \033[1;35mrr_std2: %f\033[0m' % (rr_std2_total / steps).item(),
    )

    rewards_plot.append((reward_total/steps).item())
    reward1_plot.append((rt_total1 / steps).item())
    reward2_plot.append((rr_total1 / steps).item())
    reward3_plot.append((rt_total2 / steps).item())
    reward4_plot.append((rr_total2 / steps).item())
    reward5_plot.append((rt_std1_total / steps).item())
    reward6_plot.append((rr_std1_total / steps).item())
    reward7_plot.append((rt_std2_total / steps).item())
    reward8_plot.append((rr_std2_total / steps).item())

ddpg.save()

time_end_1 = time.time()
print("运行时间：" + str(time_end_1 - time_start_1) + "秒")

plt.plot(range(0, max_episode), rewards_plot,  label='reward')
plt.plot(range(0, max_episode), reward1_plot,  label='r_t1')
plt.plot(range(0, max_episode), reward2_plot,  label='r_r1')
plt.plot(range(0, max_episode), reward3_plot,  label='r_t2')
plt.plot(range(0, max_episode), reward4_plot,  label='r_r2')
plt.plot(range(0, max_episode), reward5_plot, linestyle='--', label='rt_std1')
plt.plot(range(0, max_episode), reward6_plot, linestyle='--', label='rr_std1')
plt.plot(range(0, max_episode), reward7_plot, linestyle='--', label='rt_std2')
plt.plot(range(0, max_episode), reward8_plot, linestyle='--', label='rr_std2')


plt.legend(loc="best", fontsize=6)
plt.show()

###########################
# k=5 M=4 N=64
np.savetxt("./STARRISk05k5/m4n64/angle_T_64.txt", ris_angle_T, delimiter=',')
np.savetxt("./STARRISk05k5/m4n64/angle_R_64.txt", ris_angle_R, delimiter=',')
np.savetxt("./STARRISk05k5/m4n64/New_beta_R_64.txt", New_beta_R, delimiter=',')
np.savetxt("./STARRISk05k5/m4n64/New_beta_T_64.txt", New_beta_T, delimiter=',')
np.savetxt("./STARRISk05k5/m4n64/w_64.txt", f, delimiter=',')
savemat("./STARRISk05k5/m4n64/DDPG_64_555.mat", {"data": rewards_plot})
np.save("./STARRISk05k5/m4n64/rewards_plot_64.npy", rewards_plot)

# k=5 M=4 N=96
# np.savetxt("./STARRISk05k5/m4n96/angle_T_96.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n96/angle_R_96.txt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n96/New_beta_R_96.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n96/New_beta_T_96.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n96/w_96.txt", f, delimiter=',')
# savemat("./STARRISk05k5/m4n96/DDPG_96_555.mat", {"data": rewards_plot})
# np.save("./STARRISk05k5/m4n96/rewards_plot_96.npy", rewards_plot)

#  k=5 M=4 N=128
# np.savetxt("./STARRISk05k5/m4n128/angle_T_128.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n128/angle_R_128.txt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n128/New_beta_R_128.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n128/New_beta_T_128.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n128/w_128.txt", f, delimiter=',')
# savemat("./STARRISk05k5/m4n128/DDPG_128_555.mat", {"data": rewards_plot})
# np.save("./STARRISk05k5/m4n128/rewards_plot_128.npy", rewards_plot)

#######################
# k=5 M=8 N=64
# np.savetxt("./STARRISk05k5/m8n64/angle_T_64.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n64/angle_R_64xt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n64/New_beta_R_64.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n64/New_beta_T_64.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n64/w_64.txt", f, delimiter=',')
# savemat("./STARRISk05k5/m8n64/DDPG_64_555.mat", {"data": rewards_plot})
# np.save("./STARRISk05k5/m8n64/rewards_plot_64.npy", rewards_plot)

# k=5 M=8 N=96
# np.savetxt("./STARRISk05k5/m8n96/angle_T_96.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n96/angle_R_96.txt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n96/New_beta_R_96.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n96/New_beta_T_96.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n96/w_96.txt", f, delimiter=',')
# savemat("./STARRISk05k5/m8n96/DDPG_96_555.mat", {"data": rewards_plot})
# np.save("./STARRISk05k5/m8n96/rewards_plot_96.npy", rewards_plot)

#  k=5 M=8 N=128
# np.savetxt("./STARRISk05k5/m8n128/angle_T_128.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n128/angle_R_128.txt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n128/New_beta_R_128.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n128/New_beta_T_128.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m8n128/w_128.txt", f, delimiter=',')
# savemat("./STARRISk05k5/m8n128/DDPG_128_555.mat", {"data": rewards_plot})
# np.save("./STARRISk05k5/m8n128/rewards_plot_128.npy", rewards_plot)

######################
# k=10 M=4 N=64
# np.savetxt("./STARRISk010k10/m4n64/angle_T_64.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n64/angle_R_64.txt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n64/New_beta_R_64.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n64/New_beta_T_64.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n64/w_64.txt", f, delimiter=',')
# savemat("./STARRISk010k10/m4n64/DDPG_64_101010.mat", {"data": rewards_plot})
# np.save("./STARRISk010k10/m4n64/rewards_plot_64.npy", rewards_plot)

# k=10 M=4 N=96
# np.savetxt("./STARRISk010k10/m4n96/angle_T_96.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n96/angle_R_96.txt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n96/New_beta_R_96.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n96/New_beta_T_96.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n96/w_96.txt", f, delimiter=',')
# savemat("./STARRISk010k10/m4n96/DDPG_96_101010.mat", {"data": rewards_plot})
# np.save("./STARRISk010k10/m4n96/rewards_plot_96.npy", rewards_plot)

# k=10 M=4 N=128
# np.savetxt("./STARRISk010k10/m4n128/angle_T_128.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n128/angle_R_128.txt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n128/New_beta_R_128.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n128/New_beta_T_128.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk010k10/m4n128/w_128.txt", f, delimiter=',')
# savemat("./STARRISk010k10/m4n128/DDPG_128_101010.mat", {"data": rewards_plot})
# np.save("./STARRISk010k10/m4n128/rewards_plot_128.npy", rewards_plot)

######################
# k=10 M=8 N=64
# np.savetxt("./STARRISk010k10/m8n64/angle_T_64.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk010k10/angle_R_64.txt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk010k10/New_beta_R_64.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk010k10/New_beta_T_64.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk010k10/w_64.txt", f, delimiter=',')
# savemat("./STARRISk010k10/DDPG_64_101010.mat", {"data": rewards_plot})
# np.save("./STARRISk010k10/rewards_plot_64.npy", rewards_plot)

# k=10 M=8 N=96
# np.savetxt("./STARRISk010k10/m8n96/angle_T_96.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk010k10/m8n96/angle_R_96.txt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk010k10/m8n96/New_beta_R_96.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk010k10/m8n96/New_beta_T_96.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk010k10/m8n96/w_96.txt", f, delimiter=',')
# savemat("./STARRISk010k10/m8n96/DDPG_96_101010.mat", {"data": rewards_plot})
# np.save("./STARRISk010k10/m8n96/rewards_plot_96.npy", rewards_plot)

# k=10 M=8 N=128
# np.savetxt("./STARRISk010k10/m8n128/angle_T_128.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk010k10/m8n128/angle_R_128.txt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk010k10/m8n128/New_beta_R_128.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk010k10/m8n128/New_beta_T_128.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk010k10/m8n128/w_128.txt", f, delimiter=',')
# savemat("./STARRISk010k10/m8n128/DDPG_128_101010.mat", {"data": rewards_plot})
# np.save("./STARRISk010k10/m8n128/rewards_plot_128.npy", rewards_plot)