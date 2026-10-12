"""Summarise the selection ablation (baseline / max / min / random) from the per-episode npz dumps and stdout logs.

Usage: python scripts/analyze_selection.py logs/abl_<jobid> [--out results.md]

Per mode: success rate (Wilson 95% CI), episode length, and for the scored modes the verifier statistics.
"Change magnitude" at a step = || (a_selected - mean(initial 9 samples)) / std(initial samples) ||_2 over the 6 continuous dims
(gripper excluded: it is binarised), i.e. how far the verifier pulled the executed action from the VLA's typical action,
measured in units of the VLA's own sample spread.
"""
import argparse, glob, math, os, re
import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def episode_stats(path):
    d = np.load(path, allow_pickle=True)
    success = "success=True" in os.path.basename(path)
    scored = d["step_scored"]
    init = d["initial_actions"]
    cand, rew, sel = d["candidates"], d["rewards"], d["selected_index"]
    change, margin, rank, spread = [], [], [], []
    j = 0
    for t in range(len(scored)):
        if not scored[t]:
            continue
        a0 = np.asarray(init[t], dtype=float)
        c = np.asarray(cand[j], dtype=float)
        r = np.asarray(rew[j], dtype=float)
        s = int(sel[j]); j += 1
        mu, sd = a0.mean(0), a0.std(0)
        sd = np.where(sd < 1e-6, np.nan, sd)
        z = (c[s, :6] - mu[:6]) / sd[:6]
        change.append(float(np.sqrt(np.nansum(z ** 2))))
        order = np.argsort(-r)
        rank.append(int(np.where(order == s)[0][0]))           # 0 = highest reward
        srt = np.sort(r)[::-1]
        margin.append(float(srt[0] - srt[1]) if len(srt) > 1 else 0.0)
        spread.append(float(r.max() - r.min()))
    return dict(success=success, steps=int(d["n_steps"]), n_scored=len(change), change=change, margin=margin, rank=rank, spread=spread)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = ["| mode | episodes | successes | success rate (95% CI) | mean steps (succ / fail) |", "|---|---|---|---|---|"]
    detail = []
    for mode in ["baseline", "max", "min", "random"]:
        d = os.path.join(a.run_dir, mode)
        txt = sorted(glob.glob(os.path.join(d, "logs", "*.txt")))
        if not txt:
            continue
        log = open(txt[-1]).read()
        res = re.findall(r"^Success: (True|False)", log, flags=re.M)
        n, k = len(res), sum(r == "True" for r in res)
        lo, hi = wilson(k, n)
        eps = [episode_stats(f) for f in sorted(glob.glob(os.path.join(d, "dump", "*.npz")))]
        ss = [e["steps"] for e in eps if e["success"]]; fs = [e["steps"] for e in eps if not e["success"]]
        lines.append(f"| {mode} | {n} | {k} | {k/max(n,1):.0%} ({lo:.0%}–{hi:.0%}) | {np.mean(ss) if ss else float('nan'):.0f} / {np.mean(fs) if fs else float('nan'):.0f} |")
        if eps and mode != "baseline":
            for lab, sub in [("success", [e for e in eps if e["success"]]), ("failure", [e for e in eps if not e["success"]])]:
                ch = [x for e in sub for x in e["change"]]
                if not ch:
                    continue
                mg = [x for e in sub for x in e["margin"]]; rk = [x for e in sub for x in e["rank"]]; sp = [x for e in sub for x in e["spread"]]
                detail.append(f"| {mode} | {lab} | {len(sub)} | {np.mean(ch):.2f} | {np.median(ch):.2f} | {np.mean(mg):.2f} | {np.mean(sp):.2f} | {np.mean(rk):.1f} |")
    out = ["## Success rate", "", *lines, "", "## Verifier statistics per step (scored steps only)", "",
           "| mode | episodes | n eps | change mean (σ units) | change median | top1-top2 margin | reward spread (max-min) | rank of executed (0=top) |",
           "|---|---|---|---|---|---|---|---|", *detail]
    text = "\n".join(out)
    print(text)
    if a.out:
        open(a.out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
