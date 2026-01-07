import os, time
from pathlib import Path

ALG = "DDPG"   # TD3 那个 main 改成 "TD3"
seed = 0       # 你之后做多 seed 就改这里

# === 你想写进文件夹名的关键信息（从 env 里取也行） ===
tag = "k05k5_M8_N128_dt34_dr34"

timestamp = time.strftime("%Y%m%d-%H%M%S")
run_dir = Path("runs") / ALG / f"{tag}_seed{seed}_{timestamp}"

pth_dir = run_dir / "pth"
curve_dir = run_dir / "curves"
state_dir = run_dir / "last_state"

for d in [pth_dir, curve_dir, state_dir]:
    d.mkdir(parents=True, exist_ok=True)

print("Run dir:", run_dir)

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

max_episode = 1000
MEMORY_CAPACITY = 10000
var = 1

env = RISEnvironment()
steps = env.steps
ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path=str(pth_dir) + "/", MEMORY_CAPACITY=MEMORY_CAPACITY)
# ddpg = Net(a_dim=env.act_dim, s_dim=env.state_dim, pth_path='./pthk05k5/m4n64/', MEMORY_CAPACITY=MEMORY_CAPACITY)
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

# for i in range(max_episode):
#     reward_total = 0
#     rt_total1 = 0
#     rr_total1 = 0
#     rt_total2 = 0
#     rr_total2 = 0
#     rt_std1_total = 0
#     rr_std1_total = 0
#     rt_std2_total = 0
#     rr_std2_total = 0
#     s = env.reset()
#     # print(s)
#     for j in range(steps):
#         # 选择动作
#         a = ddpg.choose_action(s)
#         a = a.cpu().numpy()
#         a = np.clip(np.random.normal(a, var), -1, 1)
#         s_, r, rt, rr, ris_angle_T, ris_amp_T, ris_angle_R, ris_amp_R, f, New_beta_R, New_beta_T, rt_std1, rr_std1, rt_std2, rr_std2 = env.step(a)
#         # print(rt)
#         # print(rr)
#         # print("\nReward (r):", r)
#         # print("Reward for transmission user (rt):", rt)
#         # print("Reward for reflection user (rr):", rr)
#         ddpg.store_transition(s, a, r, s_)
#
#         if ddpg.pointer > MEMORY_CAPACITY:
#             var *= 0.9999
#             ddpg.learn_theta()
#         s = s_
#         reward_total += r  # 累加总奖励
#         rt_total1 += rt[0]
#         rr_total1 += rr[0]
#         rt_total2 += rt[1]
#         rr_total2 += rr[1]
#         rt_std1_total += rt_std1
#         rr_std1_total += rr_std1
#         rt_std2_total += rt_std2
#         rr_std2_total +=
start_timesteps = 10000
# explore_std = 0.2
explore_std = 1
explore_std_min = 0
explore_decay = 0.9999

global_step = 0

for i in range(max_episode):
    reward_total = 0
    rt_total1 = rr_total1 = rt_total2 = rr_total2 = 0
    rt_std1_total = rr_std1_total = rt_std2_total = rr_std2_total = 0

    s = env.reset()

    for t in range(steps):
        global_step += 1

        if global_step < start_timesteps:
            a = np.random.uniform(-1, 1, size=env.act_dim)
        else:
            a = ddpg.choose_action(s).cpu().numpy()
            a = np.clip(a + np.random.normal(0, explore_std, size=a.shape), -1, 1)

        s_, r, rt, rr, ris_angle_T, ris_amp_T, ris_angle_R, ris_amp_R, f, New_beta_R, New_beta_T, rt_std1, rr_std1, rt_std2, rr_std2 = env.step(a)
        ddpg.store_transition(s, a, r, s_)

        if global_step >= start_timesteps:
            ddpg.learn_theta()
            explore_std = max(explore_std_min, explore_std * explore_decay)

        s = s_
        reward_total += r
        rt_total1 += rt[0]; rr_total1 += rr[0]
        rt_total2 += rt[1]; rr_total2 += rr[1]
        rt_std1_total += rt_std1; rr_std1_total += rr_std1
        rt_std2_total += rt_std2; rr_std2_total += rr_std2

    # 你原来的打印/append 保持不变即可

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
plt.savefig(curve_dir / "curves.png", dpi=300, bbox_inches="tight")
plt.savefig(curve_dir / "curves.pdf", bbox_inches="tight")
plt.show()

###########################
# k=5 M=4 N=64
# np.savetxt("./STARRISk05k5/m4n64/angle_T_64.txt", ris_angle_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n64/angle_R_64.txt", ris_angle_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n64/New_beta_R_64.txt", New_beta_R, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n64/New_beta_T_64.txt", New_beta_T, delimiter=',')
# np.savetxt("./STARRISk05k5/m4n64/w_64.txt", f, delimiter=',')
# savemat("./STARRISk05k5/m4n64/DDPG_64_555.mat", {"data": rewards_plot})
# np.save("./STARRISk05k5/m4n64/rewards_plot_64.npy", rewards_plot)
np.savetxt(state_dir / "angle_T.txt", ris_angle_T, delimiter=',')
np.savetxt(state_dir / "angle_R.txt", ris_angle_R, delimiter=',')
np.savetxt(state_dir / "beta_R.txt", New_beta_R, delimiter=',')
np.savetxt(state_dir / "beta_T.txt", New_beta_T, delimiter=',')
np.savetxt(state_dir / "w_f.txt", f, delimiter=',')

np.save(curve_dir / "reward.npy", np.array(rewards_plot))
np.save(curve_dir / "rt1.npy", np.array(reward1_plot))
np.save(curve_dir / "rr1.npy", np.array(reward2_plot))
np.save(curve_dir / "rt2.npy", np.array(reward3_plot))
np.save(curve_dir / "rr2.npy", np.array(reward4_plot))
np.save(curve_dir / "rt_std1.npy", np.array(reward5_plot))
np.save(curve_dir / "rr_std1.npy", np.array(reward6_plot))
np.save(curve_dir / "rt_std2.npy", np.array(reward7_plot))
np.save(curve_dir / "rr_std2.npy", np.array(reward8_plot))

savemat(curve_dir / f"{ALG}_curves.mat", {
    "reward": np.array(rewards_plot),
    "rt1": np.array(reward1_plot),
    "rr1": np.array(reward2_plot),
    "rt2": np.array(reward3_plot),
    "rr2": np.array(reward4_plot),
    "rt_std1": np.array(reward5_plot),
    "rr_std1": np.array(reward6_plot),
    "rt_std2": np.array(reward7_plot),
    "rr_std2": np.array(reward8_plot),
})

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