"""Overlay the end-effector trajectory and the candidate actions directly on the 640x480 rollout frames.

Needs the replay file from scripts/replay_episodes.py (tcp poses, raw frames, camera) and the ablation dump (candidates, rewards).
Usage: python scripts/make_overlay_video.py logs/abl_<jobid>/<mode> <episode> [--pick rank|diverse] [--fps 5]

Drawn on the frame:  white line = real end-effector (tcp) path so far (projected with the camera params), dashed = rest of the episode,
coloured arrows = the 3 picked candidates' predicted one-step end-effector displacement.
Arrow model (fitted on the replays): displacement_world ~= 0.307 * diag(-1,-1,1) * delta_xyz  (base is yawed 180 deg; the controller only realises ~30% per step).
Arrows are magnified by ARROW_MAG so the small differences between candidates are visible.
"""
import argparse, glob, os, re
import numpy as np, imageio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("episode", type=int)
ap.add_argument("--pick", choices=["diverse", "rank"], default="rank"); ap.add_argument("--fps", type=int, default=5); ap.add_argument("--out", default=None)
ap.add_argument("--replay_dir", default="logs/replay")
a = ap.parse_args()
mode = os.path.basename(a.run_dir.rstrip("/"))
npz = [f for f in glob.glob(os.path.join(a.run_dir, "dump", "*.npz")) if re.search(rf"episode0*{a.episode}_", os.path.basename(f))][0]
d = np.load(npz, allow_pickle=True); success = "success=True" in npz
z = np.load(os.path.join(a.replay_dir, f"{mode}_ep{a.episode}.npz"))
tcp, raw, K, E = z["tcp"][:, :3], z["raw"], z["K"], z["E"]
scored = d["step_scored"]; T = len(raw)

GAIN, S, ARROW_MAG = 0.307, np.array([-1.0, -1.0, 1.0]), 12
SURF, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
LO = np.array([-0.02872725, -0.0417035, -0.02609386, -0.08092105, -0.092887, -0.20718276]); HI = np.array([0.02830968, 0.04085525, 0.04016159, 0.08192048, 0.0779285, 0.20382574])
RNG = HI - LO; names = ["x", "y", "z", "roll", "pitch", "yaw"]


def proj(P):                                         # world -> pixel (OpenCV convention; E = world->camera)
    P = np.atleast_2d(P); pc = P @ E[:3, :3].T + E[:3, 3]
    return np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], 1)


def pick3(c):
    zz = c[:, :6] / RNG; dm = np.sqrt(((zz[:, None] - zz[None]) ** 2).sum(-1))
    i, j = np.unravel_index(np.argmax(dm), dm.shape); return [int(i), int(j), int(np.argmax(np.minimum(dm[i], dm[j])))]


uv = proj(tcp)                                       # (T+1, 2) pixel path of the end effector
ymax = max(np.abs(np.asarray(c, float)[:, :6] / RNG).max() for c in d["candidates"]) * 1.08
out = a.out or f"logs/figs/{mode}_ep{a.episode}_{'success' if success else 'failure'}_overlay_{a.pick}.mp4"
os.makedirs(os.path.dirname(out), exist_ok=True)
w = imageio.get_writer(out, fps=a.fps, macro_block_size=1)
j = 0
for t in range(T):
    fig = plt.figure(figsize=(15.2, 5.2), facecolor=SURF)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.5], wspace=0.10, left=0.005, right=0.99, top=0.88, bottom=0.12)
    ax0 = fig.add_subplot(gs[0]); ax0.imshow(raw[t]); ax0.axis("off")
    ax0.plot(uv[:, 0], uv[:, 1], color="white", lw=1.2, ls=(0, (3, 3)), alpha=0.75, zorder=2)                   # whole episode (dashed)
    ax0.plot(uv[: t + 1, 0], uv[: t + 1, 1], color="black", lw=5.5, zorder=3); ax0.plot(uv[: t + 1, 0], uv[: t + 1, 1], color="white", lw=3, zorder=4)  # so far
    ax0.scatter(uv[0, 0], uv[0, 1], s=70, marker="s", color="white", edgecolor="black", linewidth=1.5, zorder=5)
    ax0.scatter(uv[t, 0], uv[t, 1], s=90, color="white", edgecolor="black", linewidth=1.8, zorder=6)
    ax = fig.add_subplot(gs[1]); ax.set_facecolor(SURF)
    if scored[t]:
        c = np.asarray(d["candidates"][j], float); r = np.asarray(d["rewards"][j], float); j += 1
        rank = np.argsort(np.argsort(-r)) + 1
        if a.pick == "diverse": idx, tags = pick3(c), ["A", "B", "C"]
        else: order = np.argsort(-r); idx, tags = [int(order[0]), int(order[len(order) // 2]), int(order[-1])], ["highest", "median", "lowest"]
        for n, i in enumerate(idx):
            end = tcp[t] + ARROW_MAG * GAIN * S * c[i, :3]; p0, p1 = uv[t], proj(end)[0]
            ax0.annotate("", xy=tuple(p1), xytext=tuple(p0), arrowprops=dict(arrowstyle="-|>", color="black", lw=5, shrinkA=0, shrinkB=0), zorder=7)
            ax0.annotate("", xy=tuple(p1), xytext=tuple(p0), arrowprops=dict(arrowstyle="-|>", color=COLORS[n], lw=2.8, shrinkA=0, shrinkB=0), zorder=8)
            ax.bar(np.arange(6) + (n - 1) * 0.27, c[i, :6] / RNG, width=0.25, color=COLORS[n], zorder=3,
                   label=f"{tags[n]}   reward {r[i]:+.2f}   rank {rank[i]}/{len(r)}" + ("  (executed)" if rank[i] == 1 else ""))
        ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK)
    ax.axhline(0, color="#c9c8c2", lw=1, zorder=2); ax.set_xticks(range(6)); ax.set_xticklabels(names, color=INK, fontsize=11); ax.set_ylim(-ymax, ymax)
    ax.set_ylabel("action / Bridge range", color=MUTED, fontsize=10); ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9c8c2")
    ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=0)
    fig.text(0.01, 0.97, f"{mode}  episode {a.episode}  step {t+1}/{T}  ->  {'SUCCESS' if success else 'FAILURE'}      " + ("highest / median / lowest reward of 32 candidates" if a.pick == "rank" else "the 3 most different of 32 candidates"), fontsize=11.5, color=INK, va="top")
    fig.text(0.01, 0.045, f"white line = real end-effector path (solid: so far, dashed: rest of episode).  Arrows = each candidate's predicted one-step end-effector motion, magnified x{ARROW_MAG} "
             f"(model: 0.31 x action, base yawed 180 deg; approximate).", fontsize=8.5, color=MUTED, va="center")
    fig.canvas.draw(); w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3]); plt.close(fig)
w.close(); print(out)
