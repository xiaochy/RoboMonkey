"""Video of one episode: left = the policy-input frame, right = the 3 most mutually different candidate actions of that step.

Usage: python scripts/make_candidate_video.py logs/abl_<jobid>/<mode> <episode_number> [--out out.mp4] [--fps 5] [--pick diverse|rank]

The 3 actions are chosen from the step's 32 Gaussian candidates by farthest-point selection in the 6-D continuous action space
(each dim divided by its Bridge action range): the most distant pair, then the candidate farthest from both.
Middle panels: the executed trajectory = cumulative sum of the executed delta-xyz (x-y top view, x-z side view), light = whole episode,
dark = so far; arrows from the current point show the 3 picked candidates (black = executed) at one-step length x ARROW_SCALE.
This is an APPROXIMATION of the end-effector path (assumes each delta is perfectly tracked, base-frame axes, not the camera frame).
Bars show each action's 6 dims / range; the legend gives the verifier reward and its rank among the 32 (rank 1 = executed).
"""
import argparse, glob, os, re
import numpy as np, imageio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("episode", type=int)
ap.add_argument("--out", default=None); ap.add_argument("--fps", type=int, default=5)
ap.add_argument("--pick", choices=["diverse", "rank"], default="diverse", help="diverse = 3 mutually most different candidates; rank = highest / median / lowest verifier reward")
a = ap.parse_args()

npz = [f for f in glob.glob(os.path.join(a.run_dir, "dump", "*.npz")) if re.search(rf"episode0*{a.episode}_", os.path.basename(f))][0]
mp4 = [f for f in glob.glob(os.path.join(a.run_dir, "rollouts", "*", "*.mp4")) if f"episode={a.episode}--" in f][0]
d = np.load(npz, allow_pickle=True)
frames = [f for f in imageio.get_reader(mp4)]
scored = d["step_scored"]; success = "success=True" in npz
assert len(frames) == len(scored), (len(frames), len(scored))

SURF, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
LO = np.array([-0.02872725, -0.0417035, -0.02609386, -0.08092105, -0.092887, -0.20718276]); HI = np.array([0.02830968, 0.04085525, 0.04016159, 0.08192048, 0.0779285, 0.20382574])
RNG = HI - LO
names = ["x", "y", "z", "roll", "pitch", "yaw"]


def pick3(c):
    z = c[:, :6] / RNG
    dm = np.sqrt(((z[:, None] - z[None]) ** 2).sum(-1))
    i, j = np.unravel_index(np.argmax(dm), dm.shape)
    k = int(np.argmax(np.minimum(dm[i], dm[j])))
    return [int(i), int(j), k]


# executed action at each step = the selected candidate; integrate its xyz to get an approximate end-effector path
ex = np.array([np.asarray(c, float)[int(k), :3] for c, k in zip(d["candidates"], d["selected_index"])])
pos = np.vstack([np.zeros(3), np.cumsum(ex, 0)])
ext = np.ptp(pos, axis=0).max() + 1e-9
ARROW_SCALE = 8                                   # arrows are drawn x8 so the small differences between candidates are visible
half = 0.62 * ext                                 # same span on every axis -> equal aspect without distorting the path
cen = (pos.min(0) + pos.max(0)) / 2
lo_p, hi_p = cen - half, cen + half
ymax = max(np.abs(np.asarray(c, float)[:, :6] / RNG).max() for c in d["candidates"]) * 1.08
out = a.out or f"logs/figs/{os.path.basename(a.run_dir.rstrip('/'))}_ep{a.episode}_{'success' if success else 'failure'}_top3{'diverse' if a.pick == 'diverse' else 'rank'}.mp4"
os.makedirs(os.path.dirname(out), exist_ok=True)
w = imageio.get_writer(out, fps=a.fps, macro_block_size=1)
j = 0
for t in range(len(scored)):
    fig = plt.figure(figsize=(16.5, 4.8), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.1, 1.5], height_ratios=[3.2, 1], wspace=0.26, hspace=0.38, left=0.01, right=0.985, top=0.84, bottom=0.11)
    ax0 = fig.add_subplot(gs[:, 0]); ax0.imshow(frames[t]); ax0.axis("off")
    axt = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]
    ax = fig.add_subplot(gs[:, 2]); ax.set_facecolor(SURF)
    if scored[t]:
        c = np.asarray(d["candidates"][j], float); r = np.asarray(d["rewards"][j], float); j += 1
        rank = np.argsort(np.argsort(-r)) + 1
        if a.pick == "diverse":
            idx = pick3(c); tags = ["A", "B", "C"]
        else:
            order = np.argsort(-r); idx = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]; tags = ["highest", "median", "lowest"]
        gripper = int(round(c[0, 6]))
        for n, i in enumerate(idx):
            ax.bar(np.arange(6) + (n - 1) * 0.27, c[i, :6] / RNG, width=0.25, color=COLORS[n], zorder=3,
                   label=f"{tags[n]}   reward {r[i]:+.2f}   rank {rank[i]}/{len(r)}" + ("  (executed)" if rank[i] == 1 else ""))
        ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK)
        ax.text(0.01, 0.02, f"gripper value = {gripper} for all three (binarised from the mean of the 9 initial samples)", transform=ax.transAxes, fontsize=9, color=MUTED)
    else:
        ax.text(0.5, 0.5, "all initial samples identical:\nverifier skipped", ha="center", va="center", transform=ax.transAxes, color=MUTED)
    axp = axt[0]; axp.set_facecolor(SURF); u, v = 0, 1
    axp.plot(pos[:, u], pos[:, v], color="#d3d2cc", lw=2.5, zorder=1)
    axp.plot(pos[: t + 1, u], pos[: t + 1, v], color=INK, lw=2.5, zorder=3)
    axp.scatter(pos[0, u], pos[0, v], s=50, color=MUTED, zorder=4, marker="s")
    axp.scatter(pos[t, u], pos[t, v], s=70, color=INK, zorder=5)
    if scored[t]:
        for n, i in enumerate(idx):
            axp.annotate("", xy=(pos[t, u] + ARROW_SCALE * c[i, u], pos[t, v] + ARROW_SCALE * c[i, v]), xytext=(pos[t, u], pos[t, v]),
                         arrowprops=dict(arrowstyle="-|>", color=COLORS[n], lw=2.6, shrinkA=0, shrinkB=0), zorder=6)
        k = int(np.argmax(r))
        axp.annotate("", xy=(pos[t, u] + ARROW_SCALE * c[k, u], pos[t, v] + ARROW_SCALE * c[k, v]), xytext=(pos[t, u], pos[t, v]),
                     arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3, ls=(0, (2, 2)), shrinkA=0, shrinkB=0), zorder=7)
    axp.set_xlim(lo_p[u], hi_p[u]); axp.set_ylim(lo_p[v], hi_p[v]); axp.set_aspect("equal", adjustable="box")
    axp.set_title(f"top view (x, y): executed path, arrows = this step's action x{ARROW_SCALE}", color=INK, fontsize=9.5, loc="left")
    axp.set_xlabel("cumulative executed delta x", color=MUTED, fontsize=8); axp.set_ylabel("cumulative executed delta y", color=MUTED, fontsize=8)
    axz = axt[1]; axz.set_facecolor(SURF)
    axz.plot(np.arange(len(pos)), pos[:, 2], color="#d3d2cc", lw=2, zorder=1); axz.plot(np.arange(t + 1), pos[: t + 1, 2], color=INK, lw=2, zorder=3)
    axz.scatter([t], [pos[t, 2]], s=50, color=INK, zorder=5); axz.set_xlim(0, len(pos) - 1)
    axz.set_title("z (cumulative executed delta) vs step", color=INK, fontsize=9, loc="left")
    for axq in (axp, axz):
        axq.tick_params(colors=MUTED, labelsize=7)
        for sp in ("top", "right"): axq.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): axq.spines[sp].set_color("#c9c8c2")
        axq.grid(color="#e6e5e0", lw=0.6, zorder=0)
    ax.axhline(0, color="#c9c8c2", lw=1, zorder=2)
    ax.set_xticks(range(6)); ax.set_xticklabels(names, color=INK, fontsize=11); ax.set_ylim(-ymax, ymax)
    ax.set_ylabel("action / Bridge range", color=MUTED, fontsize=10); ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9c8c2")
    ax.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=0)
    fig.text(0.01, 0.955, f"{os.path.basename(a.run_dir.rstrip('/'))}  episode {a.episode}  step {t+1}/{len(scored)}  ->  {'SUCCESS' if success else 'FAILURE'}      {'the 3 most different of 32 candidates' if a.pick == 'diverse' else 'highest / median / lowest reward of 32 candidates'}", fontsize=11, color=INK, va="top")
    fig.canvas.draw(); w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3]); plt.close(fig)
w.close(); print(out)
