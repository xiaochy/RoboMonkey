#!/bin/bash
#SBATCH -J vk_probe
#SBATCH -p dheld
#SBATCH -w grogu-4-8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -o /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboMonkey-reference/logs/vk_probe_%j.out
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh; conda activate simpler_env
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for d in /etc/vulkan/icd.d /usr/share/vulkan/icd.d /usr/local/share/vulkan/icd.d; do echo "== $d"; ls -la $d 2>&1; done
for f in /etc/vulkan/icd.d/*.json /usr/share/vulkan/icd.d/*.json; do [ -f "$f" ] && { echo "== $f"; cat $f; }; done
ldconfig -p | grep -E "libGLX_nvidia|libvulkan|libnvidia-glvkspirv" 
for f in "" /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
  echo "##### VK_ICD_FILENAMES='$f'"
  VK_ICD_FILENAMES=$f python - <<'PY' 2>&1 | tail -4
import sapien.core as sapien
e=sapien.Engine(); r=sapien.SapienRenderer(); print("renderer ok")
s=e.create_scene(); print("scene ok")
PY
done
