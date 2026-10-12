"""Overlay v2: end-effector trajectory + candidate actions on the 640x480 rollout frames, WITHOUT magnifying the arrows.

Differences from make_overlay_video.py (which is left untouched; its outputs are never overwritten):
  * arrows are the predicted one-step end-effector displacement at true scale (x1): 0.307 * diag(-1,-1,1) * delta_xyz, projected with the camera;
  * a zoom panel magnifies the IMAGE around the end effector (4.4x). Arrows are drawn in the same pixel coordinates there, so they stay 1:1 with the scene;
  * hollow diamond = where the end effector ACTUALLY is one step later (from the replay), for comparison with the arrows;
  * a 1 cm scale bar at the end-effector depth.

Usage: python scripts/make_overlay_video_v2.py logs/abl_<jobid>/<mode> <episode> [--pick rank|diverse] [--fps 5]
Needs logs/replay/<mode>_ep<N>.npz from scripts/replay_episodes.py.
"""
import argparse, glob, os, re, sys
import numpy as np, imageio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("episode", type=int)
ap.add_argument("--pick", choices=["diverse", "rank"], default="rank"); ap.add_argument("--fps", type=int, default=5); ap.add_argument("--out", default=None)
ap.add_argument("--replay_dir", default="logs/replay"); ap.add_argument("--half", type=int, default=36, help="half-size (px) of the zoom window")
a = ap.parse_args()
mode = os.path.basename(a.run_dir.rstrip("/"))
npz = [f for f in glob.glob(os.path.join(a.run_dir, "dump", "*.npz")) if re.search(rf"episode0*{a.episode}_", os.path.basename(f))][0]
d = np.load(npz, allow_pickle=True); success = "success=True" in npz
z = np.load(os.path.join(a.replay_dir, f"{mode}_ep{a.episode}.npz"))
tcp, raw, K, E = z["tcp"][:, :3], z["raw"], z["K"], z["E"]
scored = d["step_scored"]; T = len(raw)
out = a.out or f"logs/figs/{mode}_ep{a.episode}_{'success' if success else 'failure'}_overlayV2_{a.pick}_x1.mp4"
if os.path.exists(out):
    sys.exit(f"refusing to overwrite {out}")
os.makedirs(os.path.dirname(out), exist_ok=True)

GAIN, S = 0.307, np.array([-1.0, -1.0, 1.0])
SURF, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
LO = np.array([-0.02872725, -0.0417035, -0.02609386, -0.08092105, -0.092887, -0.20718276]); HI = np.array([0.02830968, 0.04085525, 0.04016159, 0.08192048, 0.0779285, 0.20382574])
RNG = HI - LO; names = ["x", "y", "z", "roll", "pitch", "yaw"]


def proj(P):
    P = np.atleast_2d(P); pc = P @ E[:3, :3].T + E[:3, 3]
    return np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], 1)


def pick3(c):
    zz = c[:, :6] / RNG; dm = np.sqrt(((zz[:, None] - zz[None]) ** 2).sum(-1))
    i, j = np.unravel_index(np.argmax(dm), dm.shape); return [int(i), int(j), int(np.argmax(np.minimum(dm[i], dm[j])))]


uv = proj(tcp)
ymax = max(np.abs(np.asarray(c, float)[:, :6] / RNG).max() for c in d["candidates"]) * 1.08
w = imageio.get_writer(out, fps=a.fps, macro_block_size=1)
j = 0
for t in range(T):
    fig = plt.figure(figsize=(17.6, 5.2), facecolor=SURF)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.85, 1.35], wspace=0.12, left=0.005, right=0.99, top=0.88, bottom=0.12)
    axm, axz, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    axb.set_facecolor(SURF)
    if scored[t]:
        c = np.asarray(d["candidates"][j], float); r = np.asarray(d["rewards"][j], float); j += 1
        rank = np.argsort(np.argsort(-r)) + 1
        if a.pick == "diverse": idx, tags = pick3(c), ["A", "B", "C"]
        else: order = np.argsort(-r); idx, tags = [int(order[0]), int(order[len(order) // 2]), int(order[-1])], ["highest", "median", "lowest"]
        tips = [proj(tcp[t] + GAIN * S * c[i, :3])[0] for i in idx]                     # true scale: one-step predicted displacement
        for n, i in enumerate(idx):
            axb.bar(np.arange(6) + (n - 1) * 0.27, c[i, :6] / RNG, width=0.25, color=COLORS[n], zorder=3,
                    label=f"{tags[n]}   reward {r[i]:+.2f}   rank {rank[i]}/{len(r)}" + ("  (executed)" if rank[i] == 1 else ""))
        axb.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK)
    depth = (tcp[t] @ E[:3, :3].T + E[:3, 3])[2]; bar_px = K[0, 0] * 0.01 / depth                      # 1 cm in pixels at the end-effector depth
    for ax, zoom in ((axm, False), (axz, True)):
        ax.imshow(raw[t], interpolation="bicubic" if zoom else "antialiased")
        lw = 1.0 if zoom else 1.0
        ax.plot(uv[:, 0], uv[:, 1], color="white", lw=1.2, ls=(0, (3, 3)), alpha=0.75, zorder=2)
        ax.plot(uv[: t + 1, 0], uv[: t + 1, 1], color="black", lw=5.5 if not zoom else 4.5, zorder=3); ax.plot(uv[: t + 1, 0], uv[: t + 1, 1], color="white", lw=3 if not zoom else 2.3, zorder=4)
        ax.scatter(uv[0, 0], uv[0, 1], s=70, marker="s", color="white", edgecolor="black", linewidth=1.5, zorder=5)
        ax.scatter(uv[t, 0], uv[t, 1], s=90 if not zoom else 55, color="white", edgecolor="black", linewidth=1.8, zorder=6)
        ax.scatter(uv[t + 1, 0], uv[t + 1, 1], s=70 if not zoom else 110, marker="D", facecolor="none", edgecolor="black", linewidth=1.8, zorder=9)   # actual next position
        if scored[t]:
            for n, tip in enumerate(tips):
                ax.annotate("", xy=tuple(tip), xytext=tuple(uv[t]), arrowprops=dict(arrowstyle="-|>", color="black", lw=4.2 if not zoom else 7.5, shrinkA=0, shrinkB=0, mutation_scale=10 if not zoom else 16), zorder=7)
                ax.annotate("", xy=tuple(tip), xytext=tuple(uv[t]), arrowprops=dict(arrowstyle="-|>", color=COLORS[n], lw=2.2 if not zoom else 4.2, shrinkA=0, shrinkB=0, mutation_scale=10 if not zoom else 16), zorder=8)
        if zoom:
            cx, cy = uv[t]; h = a.half
            ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy + h, cy - h)
            x0, y0 = cx - h + 5, cy + h - 6
            ax.plot([x0, x0 + bar_px], [y0, y0], color="black", lw=6, zorder=10); ax.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=3, zorder=11)
            ax.text(x0 + bar_px / 2, y0 - 2.2, "1 cm", color="white", fontsize=9, ha="center", va="bottom", zorder=12, bbox=dict(facecolor="black", alpha=0.55, pad=1.2, lw=0))
            ax.set_title(f"zoom: {2 * h}x{2 * h} px window of the 640x480 frame; arrows still 1:1 with the scene", fontsize=9, color=MUTED, loc="left")
        ax.axis("off")
    axb.axhline(0, color="#c9c8c2", lw=1, zorder=2); axb.set_xticks(range(6)); axb.set_xticklabels(names, color=INK, fontsize=11); axb.set_ylim(-ymax, ymax)
    axb.set_ylabel("action / Bridge range", color=MUTED, fontsize=10); axb.tick_params(axis="y", colors=MUTED, labelsize=8)
    for sp in ("top", "right"): axb.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): axb.spines[sp].set_color("#c9c8c2")
    axb.grid(axis="y", color="#e6e5e0", lw=0.6, zorder=0)
    fig.text(0.01, 0.97, f"{mode}  episode {a.episode}  step {t+1}/{T}  ->  {'SUCCESS' if success else 'FAILURE'}      " + ("highest / median / lowest reward of 32 candidates" if a.pick == "rank" else "the 3 most different of 32 candidates"), fontsize=11.5, color=INK, va="top")
    fig.text(0.01, 0.045, "white line = real end-effector path (solid: so far, dashed: rest).  Coloured arrow = candidate's predicted ONE-step end-effector displacement, true scale (no magnification; 0.31 x action, base yawed 180 deg, approximate).  "
             "Hollow diamond = where the end effector actually is one step later.", fontsize=8.3, color=MUTED, va="center")
    fig.canvas.draw(); w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3]); plt.close(fig)
w.close(); print(out)
