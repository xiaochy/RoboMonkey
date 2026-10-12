# RoboMonkey 复现笔记(Blackwell / grogu-4-23)

官方仓库:https://github.com/robomonkey-vla/RoboMonkey (论文 arXiv:2506.17811)
本目录是官方实现 + 三个 submodule(`openvla-mini`、`sglang-vla`、`SimplerEnv`)。

> 状态(2026-10-04):**端到端链路已跑通**。pilot:`simpler_put_eggplant_in_basket`,initial=9 / augmented=32,2 个 episode,1 成功 1 失败。
> n=2 没有统计意义,只证明链路正确。
> 正式对比实验已完成(Slurm 作业 3986934,grogu-4-8,3×RTX 3090,用时 9h56m;结果见 §4,原始数据 `logs/abl_3986934/`,汇总 `summary.md`)。
> **结论:在 eggplant、每组 30 个 episode 的设置下,没有看到 verifier 优于随机选择。**

---

## 1. 怎么理解这个 repo

RoboMonkey 是 **test-time scaling**:不重训 VLA,在推理时多采样几个动作,再用一个 verifier 打分选最好的。

```
 图像 + 指令
     │
     ▼
┌─────────────────────┐  N=initial_samples 个动作(7 个 token,temperature=1)
│ OpenVLA-7B (SGLang) │──────────────────────────────┐
│  :3200 /batch       │                              │
└─────────────────────┘                              ▼
                                      对 N 个动作按维度算 mean/var,
                                      高斯采样 M=augmented_samples 个新动作
                                      (gripper 维按 mean>=0.5 二值化,其余 clip 到 Bridge 动作范围),
                                      去重
                                                     │
                                                     ▼
┌─────────────────────┐  每个候选 → 标量 reward
│ Monkey-Verifier-7B  │◄─────────────────────────────┘
│  :3100 /process     │
└─────────────────────┘
     │  argmax reward
     ▼
 执行这一个动作(env.step)
```

三个进程:

| 进程 | 代码 | 环境 | 端口 |
|---|---|---|---|
| VLA 服务 | `sglang-vla/openvla_server.py` | `sglang-vla` | 3200 |
| Verifier 服务 | `monkey-verifier/src/infer_server.py` | `monkey-verifier` | 3100 |
| 评测客户端(仿真 + 编排) | `openvla-mini/experiments/robot/simpler/run_simpler_eval.py` | `simpler_env` | - |

客户端是**很薄**的:`get_model()` 返回 `None`,所有模型都在两个服务里。核心逻辑全在
`openvla-mini/experiments/robot/openvla_utils.py::get_vla_action`:

1. `get_batch_actions`:向 VLA 服务要 `initial_samples` 个动作(返回 7 个 token id + 反归一化后的连续动作)
2. `preprocess_actions` / `get_unique_actions`:过滤非动作 token、去重;如果只剩 1 个唯一动作,直接返回
3. `generate_augmented_samples_from_batch`:用这批动作的 **mean/var** 做高斯扰动(不是固定 sigma),再转回 token
4. `get_rewards`:把候选动作 token 发给 verifier(客户端每批 2 个,为了 24GB 卡;大显存卡可以调大)
5. `argmax` 选动作

`initial_samples=1, augmented_samples=1` 就是 baseline(直接返回 VLA 的第一个动作)。

### 1.1 Verifier 架构(`monkey-verifier/src/models/reward_model.py`)

LLaVA-RLHF 的 reward model:
- backbone:LLaVA-1.5-7B(CLIP ViT-L/336 + Vicuna-7B),QLoRA(r=64,q/k/v/o/gate/up/down),推理时 bf16
- 在 backbone 最后一个 token 的 hidden state 上接 `nn.Linear(hidden, 1)` → 标量
- 输入:当前图像 + 指令 + **7 个动作 token**。动作 token 直接用 `action_id - 1000` 映射进 Vicuna 词表
  (OpenVLA 动作 token 在词表尾部 31744–32000,减 1000 后落在 LLaMA 词表内)
- prompt 模板写死在 `RobotRewardModel.get_rewards` 里;结尾是 "the quality score of the robot action is"
- 权重:`model_dir/llava-v1.5-7b/sft_model/`(基座)+ `model_dir/lora_adapter/`(LoRA + reward_head),来自 HF `robomonkey-vla/monkey-verifier-7b`

reward 是**相对分数**,只用于同一时刻候选之间比大小(实测值如 -3 ~ -8,没有绝对意义)。

### 1.2 动作空间(对你的 bimanual 项目很重要)

- 7 维:Δxyz(3) + Δrot(3) + gripper(1),单臂 WidowX,每维离散成 256 个 bin 的 token
- 增强时的 clip 范围是 **写死的 Bridge 数据集统计量**(`openvla_utils.py::generate_augmented_samples_from_batch`)
- verifier 的输入格式、训练数据都绑定这套 7-token 动作 → **不能直接用于连续的双臂动作**。
  可借鉴的是思路(采样多个候选 → 打分 → 选最优),但 verifier 需要换动作编码并重新训练。

### 1.3 Verifier 是怎么训练的、用什么数据(训练 vs 部署)

repo **只有推理代码 + 损失函数 + 数据加载代码,没有训练脚本,也没有发布训练数据**。下面除标注"repo 核对"的以外,都来自论文
(arXiv:2506.17811;用 WebFetch 取的摘要,关键数字请对照原文)。

**训练数据:Bridge V2 的合成偏好数据**
- 源数据:Bridge V2(WidowX 真实机器人,论文写"超过 40,000 条轨迹")
- 构造:对每个状态用通用策略(OpenVLA)采样 N 个候选动作 → 聚类成 K 个代表动作 → K 选 2 两两配对,
  **按每个动作与真实专家动作的 RMSE 排序,小者为赢**。论文写总共约 2000 万对比较
- 所以 reward 学的是"这个动作离专家动作多近",**不是任务成功/进度**,也不是 per-task 微调;是一个通用 verifier
- repo 里数据格式(`data_utils_rm.py`):json,每条含 `conversations`(图像+指令)、`output_1`/`output_2`(两个候选动作)、
  `preference`(1/2)、两个动作各自的 `nrmse`

**损失(repo 核对,`models/reward_model.py::compute_loss`)**:带 margin 的 Bradley-Terry,
`-logσ( R(赢) − R(输) − α·(Δ* − Δ̂)² )`,α=0.1;Δ* 是两动作真实 NRMSE 之差,Δ̂ 是预测 reward 之差。

**训练配置(论文)**:LLaVA-1.5-7B(CLIP ViT-L)+ 线性 reward head + LoRA(论文写 rank 512 / α 128),
8×H100,batch 256,lr 2e-5,1 个 epoch。注意 repo 的 `infer_server.py` 里写的是 `lora_r=64`,与论文不一致,
没有核实是推理配置的写法还是权重本身的 rank;**复现训练前需要先确认**。

**训练和部署(评测)是不是同一个分布?——同一"家族",不是同一个数据集**

| | 数据/环境 | 与 verifier 训练数据的关系 |
|---|---|---|
| 训练 | Bridge V2 真实 WidowX 数据 | - |
| SIMPLER(我们跑的) | 仿真;真实 Bridge 场景的 real2sim 复刻 | 同一个本体、同类场景/任务;但是仿真渲染 → 图像有 sim-to-real gap;**verifier 没在仿真数据上训练,zero-shot 使用**(论文称 in-distribution) |
| 真实机器人 WidowX-250S | 真实,未见过的指令/物体/干扰物 | OOD;论文 baseline 35% → 60% |
| LIBERO-Long | 仿真,不同本体/任务 | 需要按同样流程**新建偏好数据后微调**;49.8% → 56.5% |

另外,基础策略 OpenVLA 本身也是在 Bridge V2 上预训练的,所以 SIMPLER 里 verifier 和 VLA 的训练分布是对齐的。
论文没有明确说四个 SIMPLER 任务(eggplant / stack cube / spoon / carrot)是否出现在 Bridge V2 里,只说 SIMPLER 是复刻真实场景。
论文 SIMPLER 总体 38.5% → 47.5%(+9%),各任务数字见 README。

**对 bimanual 项目的含义**:论文换本体的做法(LIBERO-Long)就是"用同样流程新建偏好数据集再微调"。
所以迁移需要:①带专家动作标注的演示数据(你们的遥操作数据可以作为"专家动作");②一个能采样候选动作的策略;
③确定动作表示(连续 14 维 or token 化)。repo 里构造偏好数据的脚本没有发布,需要自己写。

---

## 2. 环境(本机实际用的版本)

机器:`grogu-4-23`,8×RTX PRO 6000 Blackwell(sm_120,97GB)。官方 setup 脚本是为 4090 + CUDA 11.8 写的,
**直接用会失败**(torch 2.0.1+cu118 不支持 sm_120,脚本还写死 `$HOME/miniconda3` 和 sudo apt)。
所以我手动建了三个环境,偏离了官方 pin:

| 环境 | 关键版本 | 与官方的偏离 |
|---|---|---|
| `monkey-verifier` | python3.10, torch 2.7.1+cu128, transformers 4.31.0, peft 0.4.0, bitsandbytes 0.46.1, numpy 1.26.4, `setuptools<70` | torch 2.0.1+cu118 → 2.7.1+cu128;bitsandbytes 0.41.0 → 0.46.1(0.41.0 不认 CUDA 12.8);补 `setuptools<70` 提供 `pkg_resources` |
| `sglang-vla` | python3.10, torch 2.8.0+cu128, sglang 0.4.9.post3(submodule 可编辑安装), sgl-kernel 0.3.10 | torch 2.7.1 → 2.8.0;sgl-kernel 0.2.7 → 0.3.10(≤0.3.3 的 wheel 里没有 sm_120 的 rmsnorm kernel,0.3.10 开始有) |
| `simpler_env` | python3.10, numpy 1.24.4, tensorflow 2.15.0, protobuf 4.25.8, tensorflow-metadata 1.14.0, `setuptools<70`, conda-forge `libvulkan-loader` | 补 libvulkan(无 sudo);修 protobuf / tensorflow-metadata 冲突 |

没有改动的:模型权重、算法代码。安装日志在 `logs/install_*.log`、`logs/install_simpler.sh`。

### 2.1 对 submodule 代码的改动(共 2 处,均未 commit)

1. `sglang-vla/openvla_server.py`:`sgl.Runtime(...)` 增加
   `attention_backend=os.environ.get("VLA_ATTN_BACKEND") or None`、`sampling_backend=os.environ.get("VLA_SAMPLING_BACKEND") or None`。
   原因:flashinfer 要 JIT 编译,系统 nvcc 是 12.4,不支持 compute_120(试过借用另一个环境的 nvcc 12.8,又被 conda gcc 头文件冲突卡住,放弃)。
   所以运行时设 `VLA_ATTN_BACKEND=triton VLA_SAMPLING_BACKEND=pytorch`。不设则行为与官方一致。
2. `openvla-mini/experiments/robot/openvla_utils.py::get_batch_actions`:
   **上游两个仓库接口不一致**——sglang-vla 的 `/batch` 现在要 `{"instructions": [...]}`(列表),
   而 openvla-mini 客户端还在循环发 `{"instruction": ...}` 单条请求,会 422。
   改成一次请求发 `instructions=[instr]*batch_size`,与服务端对齐。

另:`.gitignore` 加了 `model_dir/`、`logs/`、`openvla-mini/transfer_images/`。

---

## 3. 怎么运行

每次开三个终端(或 nohup)。**不要 `pkill -f openvla_server`**,会把你自己的 shell 也杀掉(命令行里含这个串),用 pid kill。

```bash
cd /project_data/held/vxiao/Bimanual_Manipulation_Project/RoboMonkey-reference
source /project_data/held/vxiao/app/anaconda3/etc/profile.d/conda.sh
```

**1) Verifier(约 2–3 分钟加载,显存约 22GB)**
```bash
conda activate monkey-verifier
cd monkey-verifier/src
CUDA_VISIBLE_DEVICES=0 MODEL_DIR=$(realpath ../../model_dir) python infer_server.py   # :3100
```
自测:`POST :3100/process {"instruction","image_path","action":[[7 个 token id],...]}` → `{"rewards":[...]}`
(token id 取 31745–32000;图像是服务端可读的路径)。

**2) VLA 服务(约 1–2 分钟加载)**
```bash
conda activate sglang-vla
cd sglang-vla
VLA_ATTN_BACKEND=triton VLA_SAMPLING_BACKEND=pytorch CUDA_VISIBLE_DEVICES=1 python openvla_server.py --seed 1   # :3200
```
`curl localhost:3200/health` 返回 healthy 即可。自测:
`POST :3200/batch {"instructions":["put the eggplant in the basket"]*3,"image_path":...,"temperature":1.0}`
→ `output_ids`(7 个 token)+ `actions`(反归一化后的 7 维)。
显存约 59GB,其中大半是 SGLang 预留的 KV cache,权重只有约 15GB;
想省显存可以在 `sgl.Runtime` 里加 `mem_fraction_static`(我没试)。

**3) 评测客户端**
```bash
conda activate simpler_env
cd openvla-mini
export PRISMATIC_DATA_ROOT=. PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json     # 必须,否则 sapien 找不到 Vulkan
CUDA_VISIBLE_DEVICES=3 python experiments/robot/simpler/run_simpler_eval.py \
  --task_suite_name simpler_put_eggplant_in_basket \
  --initial_samples 9 --augmented_samples 32 \
  --num_trials_per_task 2 --local_log_dir ../logs/eval_pilot
```
- 任务:`simpler_put_eggplant_in_basket | simpler_stack_cube | simpler_spoon_on_towel | simpler_carrot_on_plate`
- `--num_trials_per_task` 默认 50(README 的数字就是 50/seed,3 个 seed);pilot 用了 2
- baseline:`--initial_samples 1 --augmented_samples 1`
- **不需要 xvfb-run**:本机上用 `xvfb-run` 反而在建 env 时 segfault(GLX 缺失);直接设 `VK_ICD_FILENAMES` 就能离屏渲染
- ⚠️ 客户端把中间图写到相对路径 `./transfer_images/`(`vla_processed_img.jpg`、`reward_img.jpg`),
  **同一个目录下同时只能跑一个客户端**,否则会互相覆盖图像。要并行评测,请在不同工作目录下启动。

### 3.1 输出
- `openvla-mini/rollouts/<日期>/…--episode=N--success=True/False--task=….mp4`:每个 episode 的回放(224×224 策略输入图,30fps,没有叠加分数)
- `--local_log_dir/EVAL-*.txt`:逐 episode 成功与否、累计成功率
- 可选 `--use_wandb True`:上传各最多 5 个成功/失败视频和按任务成功率
- 注意:官方没有 reward 分布、候选动作等可视化。如需要,要在 `get_vla_action` 里自己记录(见 §5)

---

## 4. 实验结果与速度

### 4.1 选择模式对照(eggplant,9 个初始样本 + 32 个增强候选,每组 30 个 episode,env seed 从 1000 起)

| 模式 | 怎么选 | 成功 | 成功率(95% CI) |
|---|---|---|---|
| baseline | 1/1,不用 verifier | 14/30 | 47%(30%–64%) |
| max(RoboMonkey) | verifier 最高分 | 18/30 | 60%(42%–75%) |
| min | verifier 最低分 | 13/30 | 43%(27%–61%) |
| random | 32 个候选里随机 | 19/30 | 63%(46%–78%) |

两两 Fisher 精确检验:max vs baseline p=0.44;**max vs random p=1.00**;max vs min p=0.30;random vs baseline p=0.30;random vs min p=0.20。**没有任何一对显著。**

解读(谨慎):
- max(60%) 与 random(63%) 几乎一样,**没有证据表明 verifier 的排序比随机选更好**;max/random 都高于 baseline 约 13–16 个点,
  说明增益可能主要来自"多采样后围绕样本均值取动作"(平滑效应),而不是 verifier 的排序。但 n=30 分不开,不能说 verifier 没用。
- min 最低(43%),方向上符合"最差候选更差",但同样不显著。
- 论文 README 里 eggplant:baseline 54% → 73%(3 个 seed 平均)。我们的 baseline 47% 与之同量级,max 的提升(+13)小于论文(+19),且 random 同样有提升。
  与官方设置的偏离见 §2(torch 版本、triton attention 等),不确定这些是否有影响。

### 4.2 逐步统计(`scripts/analyze_selection.py`,每步 32 个候选的 reward)

- 候选间 reward 跨度(max−min)约 3.5–4.3;第一名与第二名的差(margin)仅约 0.24 → 最高分的几个候选几乎并列。
- "改动幅度" = 被选动作相对 9 个初始样本均值的偏离,按样本标准差归一化(6 个连续维):max 2.2σ,random 2.3σ,min 3.1σ。
  max 组成功/失败 episode 的改动幅度几乎一样(2.23 vs 2.21)→ 没有看到"改动大 → 更易失败"的关联。
- reward 与"候选到 VLA 样本均值的距离"的 Spearman 相关:max −0.34,min −0.45,random −0.38(约 93–97% 的步为负)。
  即 **verifier 给分高的候选,很大程度上就是离 VLA 平均动作近的候选**。这与训练目标(离专家动作近)一致,但这里没有专家动作,
  所以只能说它偏好"共识动作",**不能据此说它排序得对不对**。
- 一个未做的对照:直接执行 9 个样本的均值(不用 verifier),看能否达到 max/random 的水平。

### 4.3 速度
- Blackwell 上 verifier 单次调用(2 个候选)约 0.10s;3090 上约 0.42s。每个决策步 32 个候选约 16 次调用 → 3090 上约 6–7s/步。
- 瓶颈是 verifier:每个候选重算约 1100 个 token(含 576 个图像 token),算力受限;批大小 2→16 仅快约 1.3 倍。
- 本次 4 组并发,约 10 小时。失败的 episode 要跑满 150 步,成功的约 30 步,所以成功率低的组更慢。

## 5. 下一步建议(未做)

1. 跑 baseline(1/1)与 RoboMonkey(9/32)各 50 episodes,同样的 env seed,比较成功率
2. 在 `get_vla_action` 里记录每步的 reward 分布、被选动作 vs 初始样本均值的差异,画 reward 曲线 / 叠加到视频,用来直观理解 verifier 在选什么
3. 想把思路迁移到 bimanual:先确定动作表示(连续 14 维?token 化?),再决定 verifier 是重训还是换成 value/critic 头

## 6. 已知坑速查

| 现象 | 原因 / 处理 |
|---|---|
| `no kernel image is available` (sgl_kernel) | sgl-kernel 版本太老,没有 sm_120;用 0.3.10 + torch 2.8 |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | flashinfer JIT;设 `VLA_ATTN_BACKEND=triton VLA_SAMPLING_BACKEND=pytorch` |
| `CUDA SETUP: Setup Failed` (bitsandbytes) | 0.41.0 不识别 CUDA 12.8;升到 0.46.1 |
| `No module named 'pkg_resources'` | `pip install "setuptools<70"` |
| `libvulkan.so.1: cannot open` | `conda install -c conda-forge libvulkan-loader`(无 sudo) |
| 建 env 时 segfault(用了 xvfb-run) | 去掉 xvfb-run,设 `VK_ICD_FILENAMES` |
| 客户端 `Field required: instructions` (422) | 上游接口不一致,见 §2.1 第 2 条 |
| protobuf `Detected incompatible Gencode/Runtime` | `--no-deps` 装 protobuf 4.25.8 + tensorflow-metadata 1.14.0 |
