#!/bin/bash
#SBATCH -J rm_median
#SBATCH -p dheld
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --exclude=grogu-4-3,grogu-4-13,grogu-4-18,grogu-4-23
#SBATCH -o /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboMonkey-reference/logs/slurm_%j.out

# "Always pick the middle-ranked candidate" on SIMPLER eggplant, only the episodes listed in EPISODES (same env seeds as the ablation).
# 2 x 24GB GPUs: verifier (:3100) on A, VLA (:3200) on B; client renders on B.
set -u
R=/project_data/held/vxiao/Bimanual_Manipulation_Project/RoboMonkey-reference
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
EPISODES=${EPISODES:-1,2,10,11}
NGPU_VISIBLE=$(nvidia-smi -L | wc -l)
if [ "$NGPU_VISIBLE" -le 2 ]; then GPUS="0,1"; else
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
IFS=',' read -r GA GB <<< "$GPUS"
[ -n "$GA" ] && [ -n "$GB" ] || { echo "could not determine 2 allocated GPUs (GPUS=$GPUS)"; exit 1; }
echo "node=$(hostname) verifier=$GA vla=$GB episodes=$EPISODES"; nvidia-smi -L
OUT=$R/logs/median_$SLURM_JOB_ID; mkdir -p $OUT
if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; else unset VK_ICD_FILENAMES; fi

conda activate simpler_env
(cd $OUT && PRISMATIC_DATA_ROOT=. PYTHONPATH=$R/openvla-mini TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=$GB python - <<'PY'
from experiments.robot.simpler.simpler_benchmark import get_benchmark
from experiments.robot.simpler.simpler_utils import get_simpler_env, get_simpler_dummy_action, get_image_from_maniskill2_obs_dict
env = get_simpler_env(get_benchmark("simpler_put_eggplant_in_basket")().get_task(0), "openvla")
obs, _ = env.reset(seed=1000); obs, *_ = env.step(get_simpler_dummy_action("openvla"))
assert get_image_from_maniskill2_obs_dict(env, obs).std() > 1.0
print("SIM_SMOKE_OK")
PY
) > $OUT/sim_smoke.log 2>&1
grep -q SIM_SMOKE_OK $OUT/sim_smoke.log || { echo "simulator smoke test FAILED"; tail -5 $OUT/sim_smoke.log; exit 1; }

conda activate monkey-verifier
(cd $R/monkey-verifier/src && CUDA_VISIBLE_DEVICES=$GA MODEL_DIR=$R/model_dir VERIFIER_PORT=3100 python infer_server.py > $OUT/verifier.log 2>&1) &
conda activate sglang-vla
(cd $R/sglang-vla && VLA_ATTN_BACKEND=triton VLA_SAMPLING_BACKEND=pytorch CUDA_VISIBLE_DEVICES=$GB python openvla_server.py --seed 1 > $OUT/vla.log 2>&1) &
for i in $(seq 1 90); do
  v=$(curl -s -m 3 localhost:3100/ | grep -c up); a=$(curl -s -m 3 localhost:3200/health | grep -c healthy)
  [ "$v$a" = "11" ] && break; sleep 10
done
[ "$v$a" = "11" ] || { echo "servers failed: verifier=$v vla=$a"; kill $(jobs -p) 2>/dev/null; exit 1; }
echo "servers up"

conda activate simpler_env
d=$OUT/median; mkdir -p $d; cd $d
PRISMATIC_DATA_ROOT=. PYTHONPATH=$R/openvla-mini TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=$GB \
python $R/openvla-mini/experiments/robot/simpler/run_simpler_eval.py --task_suite_name simpler_put_eggplant_in_basket \
  --initial_samples 9 --augmented_samples 32 --select_mode median --episode_list $EPISODES --reward_server_port 3100 --seed 7 \
  --local_log_dir $d/logs --dump_dir $d/dump > $d/stdout.log 2>&1
echo "client finished"; kill $(jobs -p) 2>/dev/null

# Post-processing (no models needed): replay the median episodes to recover tcp poses + full-res frames, then build the 3-way comparison videos.
# Needs logs/replay/{max,min}_ep<N>.npz from scripts/replay_episodes.py to exist already.
cd $R
export PRISMATIC_DATA_ROOT=. VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-} CUDA_VISIBLE_DEVICES=$GB
[ -z "$VK_ICD_FILENAMES" ] && unset VK_ICD_FILENAMES
(cd $OUT && python $R/scripts/replay_episodes.py $OUT/median $(echo $EPISODES | tr ',' ' ') --out $R/logs/replay) > $OUT/replay.log 2>&1
for ep in $(echo $EPISODES | tr ',' ' '); do
  python scripts/make_compare_video.py $ep --run highest=logs/abl_3986934/max --run median=$OUT/median --run lowest=logs/abl_3986934/min >> $OUT/compare.log 2>&1
done
echo "post-processing done"; ls logs/figs/compare_ep* 2>/dev/null
