#!/bin/bash
#SBATCH -J robomonkey_abl
#SBATCH -p dheld
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=14:00:00
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-4-18,grogu-4-23
#SBATCH -o /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboMonkey-reference/logs/slurm_%j.out

# Selection ablation on SIMPLER eggplant: baseline (1/1), max / min / random over 9+32 candidates, 30 episodes each.
# 3 x 24GB GPUs (RTX 3090 on grogu-4-8; 4-13 has 12GB 3080Ti and is excluded, the verifier needs ~21GB): verifier1 (:3100) on A, verifier2 (:3101) on B, VLA (:3200) on C. Clients render on C.
set -u
R=/project_data/held/vxiao/Bimanual_Manipulation_Project/RoboMonkey-reference
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
N=${N_EPISODES:-30}
# GPUs allocated to this job. Nodes with cgroup isolation renumber them 0..n-1; others expose all GPUs, so use the physical IDX.
NGPU_VISIBLE=$(nvidia-smi -L | wc -l)
if [ "$NGPU_VISIBLE" -le 3 ]; then
  GPUS="0,1,2"
else
  IDX=$(scontrol show job -d $SLURM_JOB_ID | grep -o "IDX:[0-9,-]*" | head -1 | cut -d: -f2)
  GPUS=$(python - "$IDX" <<'PY'
import sys
out=[]
for part in sys.argv[1].split(","):
    if "-" in part:
        a,b=part.split("-"); out+=list(range(int(a),int(b)+1))
    elif part: out.append(int(part))
print(",".join(map(str,out)))
PY
)
fi
IFS=',' read -r GA GB GC <<< "$GPUS"
[ -n "$GA" ] && [ -n "$GB" ] && [ -n "$GC" ] || { echo "could not determine 3 allocated GPUs (GPUS=$GPUS)"; exit 1; }
echo "node=$(hostname) verifier1=$GA verifier2=$GB vla=$GC"; nvidia-smi -L
OUT=$R/logs/abl_$SLURM_JOB_ID; mkdir -p $OUT

# Vulkan ICD differs per node: only pin it when the file exists (grogu-4-23); otherwise use the default search path (grogu-4-8)
if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; else unset VK_ICD_FILENAMES; fi

# Fail fast: simulator smoke test (create env, reset, one step, check a non-blank frame) before loading any model
conda activate simpler_env
(cd $OUT && PRISMATIC_DATA_ROOT=. PYTHONPATH=$R/openvla-mini TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=$GC python - <<'PY'
import numpy as np
from experiments.robot.simpler.simpler_benchmark import get_benchmark
from experiments.robot.simpler.simpler_utils import get_simpler_env, get_simpler_dummy_action, get_image_from_maniskill2_obs_dict
env = get_simpler_env(get_benchmark("simpler_put_eggplant_in_basket")().get_task(0), "openvla")
obs, _ = env.reset(seed=1000)
obs, *_ = env.step(get_simpler_dummy_action("openvla"))
img = get_image_from_maniskill2_obs_dict(env, obs)
assert img.std() > 1.0, "rendered frame is blank"
print("SIM_SMOKE_OK", img.shape)
PY
) > $OUT/sim_smoke.log 2>&1
grep -q SIM_SMOKE_OK $OUT/sim_smoke.log || { echo "simulator smoke test FAILED"; tail -5 $OUT/sim_smoke.log; exit 1; }
echo "simulator smoke test ok"

conda activate monkey-verifier
(cd $R/monkey-verifier/src && CUDA_VISIBLE_DEVICES=$GA MODEL_DIR=$R/model_dir VERIFIER_PORT=3100 python infer_server.py > $OUT/verifier1.log 2>&1) &
(cd $R/monkey-verifier/src && CUDA_VISIBLE_DEVICES=$GB MODEL_DIR=$R/model_dir VERIFIER_PORT=3101 python infer_server.py > $OUT/verifier2.log 2>&1) &
conda activate sglang-vla
(cd $R/sglang-vla && VLA_ATTN_BACKEND=triton VLA_SAMPLING_BACKEND=pytorch CUDA_VISIBLE_DEVICES=$GC python openvla_server.py --seed 1 > $OUT/vla.log 2>&1) &

echo "waiting for servers..."
for i in $(seq 1 90); do
  v1=$(curl -s -m 3 localhost:3100/ | grep -c up); v2=$(curl -s -m 3 localhost:3101/ | grep -c up); a=$(curl -s -m 3 localhost:3200/health | grep -c healthy)
  [ "$v1$v2$a" = "111" ] && break; sleep 10
done
[ "$v1$v2$a" = "111" ] || { echo "servers failed to start: v1=$v1 v2=$v2 vla=$a"; kill $(jobs -p) 2>/dev/null; exit 1; }
echo "servers up"

conda activate simpler_env
launch() { # name init aug mode reward_port
  d=$OUT/$1; mkdir -p $d; cd $d
  PRISMATIC_DATA_ROOT=. PYTHONPATH=$R/openvla-mini TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=$GC \
  python $R/openvla-mini/experiments/robot/simpler/run_simpler_eval.py --task_suite_name simpler_put_eggplant_in_basket \
    --initial_samples $2 --augmented_samples $3 --select_mode $4 --reward_server_port $5 --num_trials_per_task $N --seed 7 \
    --local_log_dir $d/logs --dump_dir $d/dump > $d/stdout.log 2>&1 &
}
launch baseline 1 1 max 3100
launch max 9 32 max 3100
launch min 9 32 min 3101
launch random 9 32 random 3101
wait $(jobs -p | tail -4)   # wait for the 4 clients only
echo "clients finished"; kill $(jobs -p) 2>/dev/null
