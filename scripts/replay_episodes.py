"""Replay saved episodes (same env seed + the executed actions) in SIMPLER to recover what the ablation did not store:
the end-effector (tcp) pose per step, the full-resolution 3rd-person frames and the camera parameters.
Also checks that the replay reproduces the original run (success flag, and frames vs the saved 224x224 rollout mp4).

Usage (simpler_env, from a scratch cwd):  python scripts/replay_episodes.py logs/abl_<jobid>/<mode> 1 10 --out logs/replay
"""
import argparse, glob, os, re, sys, itertools
import numpy as np, imageio
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "openvla-mini"))
from experiments.robot.simpler.simpler_benchmark import get_benchmark
from experiments.robot.simpler.simpler_utils import get_simpler_env, get_simpler_img, convert_maniskill, get_image_from_maniskill2_obs_dict

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("episodes", type=int, nargs="+"); ap.add_argument("--out", default="logs/replay")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
env = get_simpler_env(get_benchmark("simpler_put_eggplant_in_basket")().get_task(0), "openvla")
mode = os.path.basename(a.run_dir.rstrip("/"))
for ep in a.episodes:
    npz = [f for f in glob.glob(os.path.join(a.run_dir, "dump", "*.npz")) if re.search(rf"episode0*{ep}_", os.path.basename(f))][0]
    d = np.load(npz, allow_pickle=True); orig_success = "success=True" in npz
    acts = np.array([np.asarray(c, float)[int(k)] for c, k in zip(d["candidates"], d["selected_index"])])      # executed 7-d actions
    saved = [f for f in imageio.get_reader(glob.glob(os.path.join(a.run_dir, "rollouts", "*", f"*episode={ep}--*.mp4"))[0])]
    obs, _ = env.reset(seed=1000 + ep - 1)                                                                       # seeds = count(1000), episode k -> 1000+k-1
    cam = obs["camera_param"]["3rd_view_camera"]
    tcp, raw, diffs, done = [obs["extra"]["tcp_pose"].copy()], [], [], False
    for t, act in enumerate(acts):
        raw.append(get_image_from_maniskill2_obs_dict(env, obs).copy())
        small = get_simpler_img(env, obs, 224)                                   # same preprocessing as the original run
        diffs.append(float(np.abs(small.astype(float) - saved[t].astype(float)).mean()))
        obs, rew, done, trunc, info = env.step(convert_maniskill(act))
        tcp.append(obs["extra"]["tcp_pose"].copy())
        if done: break
    base = env.unwrapped.agent.robot.pose
    np.savez_compressed(os.path.join(a.out, f"{mode}_ep{ep}.npz"), tcp=np.array(tcp), raw=np.array(raw), acts=acts, K=np.array(cam["intrinsic_cv"]), E=np.array(cam["extrinsic_cv"]),
                        base_p=np.array(base.p), base_q=np.array(base.q), done=bool(done), orig_success=orig_success, frame_mad=np.array(diffs))
    print(f"ep{ep}: steps replayed={len(raw)}/{len(acts)}  replay success={bool(done)} original={orig_success}  mean|frame diff| vs saved mp4: mean={np.mean(diffs):.2f} max={np.max(diffs):.2f} (0-255 scale)", flush=True)
