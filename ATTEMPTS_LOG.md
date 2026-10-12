# RoboMonkey 复现:尝试记录(简版)

截止 2026-10-06。详细的使用说明、架构和踩坑见 `RoboMonkey_NOTES.md`,本文只记"做过什么、结果如何、还没做什么"。

## 1. 环境与跑通

- 拉取 submodule、下载权重(verifier 16.6GB、OpenVLA-7B 15GB),建 3 个 conda 环境:`monkey-verifier` / `sglang-vla` / `simpler_env`。
- 官方脚本为 4090 + CUDA 11.8 写的,在 Blackwell(sm_120)和无 sudo 的环境上不能直接用。主要修复:
  torch 换成 cu128(verifier 2.7.1,sglang 2.8.0)、bitsandbytes 升到 0.46.1、sgl-kernel 升到 0.3.10(旧版没有 sm_120 kernel)、
  `setuptools<70`、protobuf/tensorflow-metadata 版本、conda-forge 装 `libvulkan-loader`、flashinfer 的 JIT 因系统 nvcc 12.4 失败 → 改用 triton attention。
- 代码改动(未 commit):`openvla_server.py` 加后端/显存环境变量;`openvla_utils.py` 客户端对齐服务端的批量接口(上游两个仓库接口不一致,会 422)、
  加 `select_mode`(max/min/random)、`reward_batch_size`、逐步日志;`run_simpler_eval.py` 加对应参数和 npz 导出。
- Pilot(eggplant,9+32,2 个 episode):1/2 成功,只证明链路通。

## 2. 走过的弯路(如实记录)

- **用错了 GPU**:先在 grogu-4-23 上手动指定 GPU 0/1/3 跑,但作业只分到 GPU 2;其中 GPU 1 上还有别人的训练任务。发现后全部停掉,那批数据作废(`logs/runs_discarded/`)。
- **批大小提速没用**:以为把 verifier 批大小从 2 调到 16 能快 8–16 倍,实测只快约 1.3 倍(算力受限),且数值有最大约 0.9 的差异,所以保持官方默认 2。
- **Slurm 作业几次失败**:3986922 取消(物理 GPU 编号与作业内编号不一致,且 grogu-4-8 是 24GB 的 3090);3986924 服务正常但仿真客户端 segfault(grogu-4-8 的 Vulkan ICD 路径与 4-23 不同,
  不设 `VK_ICD_FILENAMES` 即可;之后加了仿真冒烟测试,失败即退出);3986928 被调度到 grogu-4-13(12GB 3080 Ti)后取消。
- **抢占**:3986934 提交前 `--test-only` 显示会抢占 sjaiswa3 的作业,征得你同意后才提交。
- 最终成功的作业:**3986934**(grogu-4-8,3×3090:verifier ×2 + VLA ×1),用时 9h56m。

## 3. 对照实验(eggplant,每组 30 个 episode,env seed 从 1000 起)

| 组 | 做法 | 成功 | 成功率(95% CI) |
|---|---|---|---|
| baseline | 1/1,不用 verifier | 14/30 | 47%(30%–64%) |
| max | 9+32,verifier 最高分 | 18/30 | 60%(42%–75%) |
| min | 9+32,verifier 最低分 | 13/30 | 43%(27%–61%) |
| random | 9+32,随机选 | 19/30 | 63%(46%–78%) |

两两 Fisher 检验均不显著(max vs random p=1.00)。**这个设置下没有看到 verifier 优于随机选择**;增益更可能来自"采样后围绕均值取动作"。n=30、单 seed,不能说 verifier 没用。
与论文偏离:环境版本不同;论文 baseline 54% → 73%,我们 47% → 60%,且 random 同样有提升。

## 4. 分析(基于 7491 个决策步的 npz)

- **候选多样性**:没有显式的多样性指标。每步用 9 个 VLA 样本拟合**一个对角协方差的 6 维高斯**(夹爪维用均值阈值二值化)再采 32 个;
  32 个候选里**不包含**原来的 9 个。候选散布与初始样本几乎一致(标准差比 0.94–0.98),旋转维散布(8%–12% 动作范围)大于平移维(4%–6.5%)。
  高斯会把多峰样本抹成单峰、丢掉维度间相关性,夹爪的不确定性被消除。
- **reward 偏好离均值近的候选**:reward 与"到 9 个样本均值的距离"的 Spearman 相关为 -0.34(max)/-0.45(min)/-0.38(random)。
  与"改动幅度"一致:min 3.1σ、max 2.2σ、random 2.3σ。max 组成功/失败 episode 的改动幅度几乎一样,没看到关联。
  没有专家动作,所以不能说排序"对不对",只能说它偏好"共识动作"。
- **速度**:瓶颈是 verifier(每个候选重算约 1100 个 token);单次调用 Blackwell 约 0.10s,3090 约 0.42s;3090 上约 6–7s/步。
- **训练数据/损失**(论文 + repo 核对):合成偏好数据来自 Bridge V2(约 2000 万对,按候选与专家动作的 RMSE 排序),带 margin 的 Bradley-Terry(α=0.1)。
  repo 里**没有训练脚本和数据**;LoRA rank 论文写 512,repo 推理配置写 64,未核实。

## 5. 可视化

- 静态图:一个决策步里 9 个初始样本 vs 32 个候选(`logs/figs/initial_vs_candidates_example.png`)。
- 侧面板视频(`*_top3diverse/rank.mp4`):画面 + x–y 路径 + z 曲线 + 3 个候选的柱状图。轨迹是累加执行动作得到的近似。
- **叠加视频**(`*_overlay_rank/diverse.mp4`):把末端轨迹和候选箭头直接投影到 640×480 的 rollout 画面上。做法:用存下的 env seed 和执行动作在仿真里**重放**
  episode(成功/失败与原来一致,画面与原 mp4 平均差 2.5/4.5),取回 tcp 位姿、原始画面和相机外参/内参;箭头用拟合的"位移 ≈ 0.307 × diag(-1,-1,1) × Δxyz"(R²=0.33)换算后投影。
  箭头只含平移,放大 12 倍;只做了目视检查,未做定量重投影误差。

- **叠加视频 v2(不放大)**(`*_overlayV2_*_x1.mp4`,脚本 `make_overlay_video_v2.py`,旧文件未覆盖):箭头按真实比例(预期一步位移,0.31×动作),
  另加画面局部放大图、夹爪下一步真实位置(空心菱形)、1 厘米比例尺。覆盖 max 的 episode 1/2/10/11 和 min 的 episode 1;新增 episode 的重放同样复现了原成败(画面差 2.6–5.1)。
  局限:夹爪靠近画面边缘时局部放大图会有一部分是空白。

## 6. 还没做 / 待定

- 加一组"直接执行 9 个样本的均值"的对照,检验增益是否来自平滑。
- 多 seed 或更多 episode;其他 3 个 SIMPLER 任务。
- 官方 verifier 零样本在其他数据上的排序相关性(判断能否迁移)。
- 提速:共享前缀 KV cache(只在读代码层面判断过,未实现)。
- 迁移到 RoboTwin(双臂):只讨论了设计(连续动作编码、复用场景 token、逐臂归一化、先做离线排序检验),没有实现;阶段 1 用哪个单臂 benchmark 等你决定。

## 7. 文件索引

| 内容 | 位置 |
|---|---|
| 使用说明、架构、踩坑 | `RoboMonkey_NOTES.md` |
| 实验原始数据与汇总 | `logs/abl_3986934/`(`summary.md`、各组 `dump/*.npz`、`logs/`、`rollouts/`) |
| 作废数据 | `logs/runs_discarded/` |
| 提交脚本 | `scripts/run_selection_ablation.sbatch` |
| 汇总分析 | `scripts/analyze_selection.py` |
| 重放 / 视频 | `scripts/replay_episodes.py`、`scripts/make_candidate_video.py`、`scripts/make_overlay_video.py` |
| 视频与图 | `logs/figs/` |
