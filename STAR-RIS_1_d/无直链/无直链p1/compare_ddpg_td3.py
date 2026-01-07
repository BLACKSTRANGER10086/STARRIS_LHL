import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
def find_latest_run(alg: str, tag_contains: str = "") -> Path:
    base = Path("runs") / alg
    assert base.exists(), f"Not found: {base}"
    runs = [p for p in base.iterdir() if p.is_dir() and tag_contains in p.name]
    assert len(runs) > 0, f"No runs found under {base} with tag_contains='{tag_contains}'"
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]

def load_reward(run_dir: Path) -> np.ndarray:
    p = run_dir / "curves" / "reward.npy"
    assert p.exists(), f"Not found: {p}"
    return np.load(p)

def moving_avg(x: np.ndarray, w: int = 20) -> np.ndarray:
    if w <= 1:
        return x
    w = min(w, len(x))
    return np.convolve(x, np.ones(w)/w, mode="valid")

# === 你只需要改这两个：要对比的 run 文件夹 ===
# 方式1：自动找最新一次（并按 tag_contains 过滤）
ddpg_run = find_latest_run("DDPG", tag_contains="k05k5_M8_N128_dt34_dr34")
td3_run  = find_latest_run("TD3",  tag_contains="k05k5_M8_N128_dt34_dr34")

# 方式2：你也可以手动写死路径（更稳）
# ddpg_run = Path(r"runs\DDPG\k05k5_M8_N128_dt34_dr34_seed0_20251231-151001")
# td3_run  = Path(r"runs\TD3\k05k5_M8_N128_dt34_dr34_seed0_20251231-103107")

ddpg_r = load_reward(ddpg_run)
td3_r  = load_reward(td3_run)

# 对齐长度（避免一个跑1000一个跑400）
L = min(len(ddpg_r), len(td3_r))
# L = 500
ddpg_r = ddpg_r[:L]
td3_r  = td3_r[:L]

print("DDPG run:", ddpg_run)
print("TD3  run:", td3_run)
print("Episodes compared:", L)

# 画原始曲线
plt.figure()
plt.plot(ddpg_r, label="DDPG r")
plt.plot(td3_r,  label="TD3 r")

# 画平滑曲线（可选）
# w = 20
# ddpg_ma = moving_avg(ddpg_r, w)
# td3_ma  = moving_avg(td3_r,  w)
# plt.plot(np.arange(len(ddpg_ma)) + (w-1), ddpg_ma, linestyle="--", label=f"DDPG r MA{w}")
# plt.plot(np.arange(len(td3_ma))  + (w-1), td3_ma,  linestyle="--", label=f"TD3 r MA{w}")

plt.xlabel("Episode")
plt.ylabel("r (avg per episode)")
plt.legend()
plt.tight_layout()

save_dir = Path("runs") / "compare"
save_dir.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = save_dir / f"compare_r_{ts}.png"

plt.savefig(save_path, dpi=300, bbox_inches="tight")
print("Saved figure to:", save_path)
plt.show()
