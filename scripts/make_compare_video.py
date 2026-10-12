"""Three rollouts of the SAME episode (same env seed => same initial scene) side by side, plus all three end-effector paths overlaid on one frame.

Usage: python scripts/make_compare_video.py <episode> --run highest=logs/abl_3986934/max --run median=logs/median_<job>/median --run lowest=logs/abl_3986934/min
Needs logs/replay/<basename(run_dir)>_ep<episode>.npz for every run (scripts/replay_episodes.py).
Top row: the three rollouts (full-res frames) with their own end-effector path so far. Bottom-left: the three paths drawn on the common initial frame, growing with time.
Rollouts that end earlier freeze on their last frame and show the outcome. Output never overwrites an existing file.
"""
import argparse, glob, os, re, sys
import numpy as np, imageio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("episode", type=int)
ap.add_argument("--run", action="append", required=True, help="label=run_dir (3 of them, in display order)")
ap.add_argument("--replay_dir", default="logs/replay"); ap.add_argument("--fps", type=int, default=5); ap.add_argument("--out", default=None)
a = ap.parse_args()
runs = [r.split("=", 1) for r in a.run]
COLORS = {"highest": "#2a78d6", "median": "#eb6834", "lowest": "#1baf7a"}
SURF, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
data = []
for label, rd in runs:
    mode = os.path.basename(rd.rstrip("/"))
    z = np.load(os.path.join(a.replay_dir, f"{mode}_ep{a.episode}.npz"))
    npz = [f for f in glob.glob(os.path.join(rd, "dump", "*.npz")) if re.search(rf"episode0*{a.episode}_", os.path.basename(f))][0]
    data.append(dict(label=label, mode=mode, tcp=z["tcp"][:, :3], raw=z["raw"], K=z["K"], E=z["E"], ok="success=True" in npz, n=len(z["raw"])))
for d in data[1:]:
    assert np.allclose(d["E"], data[0]["E"]) and np.allclose(d["K"], data[0]["K"]), "cameras differ"
    assert np.abs(d["raw"][0].astype(float) - data[0]["raw"][0].astype(float)).mean() < 3, "initial scenes differ (different seed?)"
K, E = data[0]["K"], data[0]["E"]
def proj(P):
    P = np.atleast_2d(P); pc = P @ E[:3, :3].T + E[:3, 3]
    return np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], 1)
for d in data:
    d["uv"] = proj(d["tcp"]); d["zc"] = (d["tcp"] @ E[:3, :3].T + E[:3, 3])[:, 2]
    bad = (d["zc"] < 0.05) | (d["uv"][:, 0] < 0) | (d["uv"][:, 0] > 640) | (d["uv"][:, 1] < 0) | (d["uv"][:, 1] > 480)   # behind the camera / outside the frame
    d["bad"] = bad; d["uv"][bad] = np.nan; d["len_m"] = float(np.linalg.norm(np.diff(d["tcp"], axis=0), axis=1).sum())
T = max(d["n"] for d in data)
out = a.out or f"logs/figs/compare_ep{a.episode}_{'_'.join(d['label'] for d in data)}.mp4"
if os.path.exists(out): sys.exit(f"refusing to overwrite {out}")
os.makedirs(os.path.dirname(out), exist_ok=True)
w = imageio.get_writer(out, fps=a.fps, macro_block_size=1)
init = data[0]["raw"][0]
for t in range(T):
    fig = plt.figure(figsize=(18, 9.6), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.12, wspace=0.04, left=0.005, right=0.995, top=0.93, bottom=0.01)
    for k, d in enumerate(data):
        ax = fig.add_subplot(gs[0, k]); tt = min(t, d["n"] - 1); col = COLORS.get(d["label"], "#888888")
        ax.imshow(d["raw"][tt]); ax.axis("off")
        ax.plot(d["uv"][: tt + 1, 0], d["uv"][: tt + 1, 1], color="black", lw=5, zorder=3); ax.plot(d["uv"][: tt + 1, 0], d["uv"][: tt + 1, 1], color=col, lw=2.8, zorder=4)
        ax.scatter(d["uv"][tt, 0], d["uv"][tt, 1], s=80, color="white", edgecolor="black", linewidth=1.8, zorder=6)
        ax.set_xlim(0, 640); ax.set_ylim(480, 0)
        if d["bad"][tt]: ax.text(320, 455, "end effector out of the camera view", color="white", fontsize=11, ha="center", bbox=dict(facecolor="black", alpha=0.65, pad=3, lw=0))
        done = t >= d["n"] - 1
        ax.set_title(f"{d['label']} reward every step   step {tt + 1}/{d['n']}" + (f"   ENDED: {'SUCCESS' if d['ok'] else 'FAILURE'}" if done else ""), color=col if not done else INK, fontsize=12, loc="left", fontweight="bold")
    axo = fig.add_subplot(gs[1, :2]); axo.imshow(init); axo.axis("off")
    for d in data:
        col = COLORS.get(d["label"], "#888888"); tt = min(t, d["n"] - 1)
        axo.plot(d["uv"][: tt + 1, 0], d["uv"][: tt + 1, 1], color="black", lw=6, zorder=3); axo.plot(d["uv"][: tt + 1, 0], d["uv"][: tt + 1, 1], color=col, lw=3.4, zorder=4, label=d["label"])
        axo.scatter(d["uv"][tt, 0], d["uv"][tt, 1], s=110, color=col, edgecolor="black", linewidth=1.8, zorder=6)
    axo.set_xlim(0, 640); axo.set_ylim(480, 0)
    axo.scatter(data[0]["uv"][0, 0], data[0]["uv"][0, 1], s=90, marker="s", color="white", edgecolor="black", linewidth=1.6, zorder=7)
    axo.set_title("end-effector paths of the three rollouts on the common initial frame (same env seed)", color=INK, fontsize=11, loc="left")
    axo.legend(loc="lower left", frameon=True, facecolor=SURF, edgecolor="none", fontsize=11)
    axt = fig.add_subplot(gs[1, 2]); axt.axis("off")
    rows = [f"{'':9s}{'outcome':>10s}{'steps':>7s}{'path (m)':>10s}"] + [f"{d['label']:9s}{('SUCCESS' if d['ok'] else 'FAILURE'):>10s}{d['n']:>7d}{d['len_m']:>10.2f}" for d in data]
    axt.text(0.0, 0.9, "\n".join(rows), family="monospace", fontsize=12, color=INK, va="top")
    axt.text(0.0, 0.42, "Same episode seed => identical starting scene.\nEach rollout picks, at every step, the candidate\nwith the highest / median / lowest verifier reward\namong 32 Gaussian candidates.\nPath length = sum of end-effector movement (world, m).\nPaths are cut where the end effector leaves the camera view.\nWhite square = start; dots = current position.", fontsize=10.5, color=MUTED, va="top")
    fig.text(0.01, 0.975, f"episode {a.episode}   t = {t + 1}/{T}", fontsize=13, color=INK, va="center")
    fig.canvas.draw(); w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3]); plt.close(fig)
w.close(); print(out)
