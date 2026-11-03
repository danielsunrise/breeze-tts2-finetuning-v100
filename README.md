# Breeze TTS 2 · V100 训练适配

基于 [instavar/breeze-tts2-finetuning](https://github.com/instavar/breeze-tts2-finetuning) 的独立 V100 / Volta 适配版本，支持 LoRA 和全量 SFT。训练方案来自社区，**不是 BreezeBlue 官方训练代码**。推理部署另见 [breeze-tts-v100](https://github.com/danielsunrise/breeze-tts-v100)。

社区基线：`7d87c54fbf346cfe31d9786aad5a2918e08f1c7d`。保留原 [README.community.md](README.community.md)、[LICENSE](LICENSE)、[NOTICE](NOTICE)、[UPSTREAM.md](UPSTREAM.md) 和研究来源说明。仓库只分发源码和通用配置，不分发底模、音色权重、数据或日志。

## V100 改动

- V100 无原生 BF16，合成主干改用 FP16；文本编码器和文本投影保留 FP32，以避免已观察到的纯 FP16 溢出。
- 文本编码器初始化继承加载 dtype，投影输出转换回合成端 dtype。
- FP16 底模上的 LoRA 参数保持 FP32，避免 AdamW 状态和 epsilon 下溢；前向时进行必要的 dtype 转换。
- 使用非重入梯度检查点，确保冻结底模后适配器仍能获得梯度。
- LoRA 使用 GradScaler，checkpoint 保存和恢复其状态；每十步报告进度。
- 增加 GPT-SoVITS `.list` 转换、完整清单缓存准备、限步/限轮分阶段 LoRA 工具。底模直接复用已有目录，无需复制到训练仓库。

## 已验证范围

原部署在 Tesla V100-SXM2 16GB、Python 3.12、PyTorch 2.9.1+cu126 / torchaudio 2.9.1+cu126 上验证：

- LoRA 前向、反向、有限且非零梯度、优化器更新，以及新进程 checkpoint 恢复、适配器重新加载。
- 实际完成过受步数和样本预算约束的短轮次 LoRA 试训。
- 两秒短音频部署测试中，LoRA 峰值已分配约 8.64 GiB、预留约 9.76 GiB；不代表长音频容量。
- 全量 SFT 的默认 FP32-master SGD 在 16GB 上显存不足；社区自带 Adafactor 完成短音频的单步前向、反向、更新及验证，峰值已分配约 12.44 GiB、预留约 12.83 GiB。

**全量训练只验证了短音频单步计算**，没有在 V100 上验证持续收敛、长片段或完整全量 checkpoint 的保存恢复。公开版的路径与预算辅助工具本次通过 CPU/模拟训练回归，没有为发布重新启动 GPU 训练。

## GPU 兼容范围

**V100 已实测适配；其他老卡尚未逐卡验证。本项目不承诺所有不支持原生 BF16 的显卡都能运行。**

代码会在 CUDA 设备不支持原生 BF16 时选择 FP16，并对部分模块保留 FP32；支持原生 BF16 的设备继续使用 BF16 分支。这只处理精度选择和已发现的稳定性问题，不会自动补齐驱动、框架、算子或显存方面的限制。

| 显卡 / 架构 | 当前验证状态 |
| --- | --- |
| V100 / Volta（`sm_70`） | 原部署已验证 LoRA 更新、checkpoint 保存/恢复和受预算约束的试训；全量 SFT 仅验证短音频 Adafactor 单步计算。 |
| T4 等 Turing（`sm_75`） | 未做硬件实测；满足软件、算子及显存条件时可尝试 FP16 路径 |
| P100 / P40 等 Pascal（`sm_60` / `sm_61`） | 未做硬件实测；需要验证算子、FP16 性能与数值稳定性 |
| Maxwell（`sm_5x`） | 未做硬件实测，不提供兼容保证 |
| K80 等 Kepler（`sm_37`）及更老架构 | 不在本项目当前 PyTorch 2.9.1+cu126 环境的支持范围内 |

即使 PyTorch 构建包含目标架构，也不代表 Breeze 的全部依赖和流程都已通过验证。使用前仍需确认：

1. **依赖支持**：PyTorch、CUDA 运行库、驱动及相关算子都支持该 GPU。当前验证环境为 `torch==2.9.1+cu126`；不要把任意最新版依赖当作等价替换。NVIDIA 列出的 Maxwell / Pascal / Volta 工具链支持止于 CUDA 12.x，Kepler 止于更早版本。
2. **容量足够**：FP32 模块和运行缓存仍会占显存。V100 原部署的推理常驻约 11 GiB，仅为该配置的参考；6GB、8GB 显卡不能直接照搬这个常驻配置。
3. **逐流程验证**：先检查有限值和短音频推理，再测试 LoRA 梯度、保存恢复及实际显存。推理成功不等于 LoRA 或全量训练也能成功；全量训练的显存需求和稳定性需单独验证。

可查看本机安装的构建，而不是只看显卡是否支持 BF16：

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Compiled architectures:", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("Native BF16:", torch.cuda.is_bf16_supported(including_emulation=False))
```

`nvidia-smi` 显示的 CUDA 版本是驱动支持的最高版本，不等于 PyTorch 实际使用的 CUDA 运行库；后者看 `torch.version.cuda`。较新 GPU 也要确认所装构建覆盖其架构，不能仅凭存在 BF16 分支就认为全部兼容。

参考：[NVIDIA GPU 架构表](https://developer.nvidia.com/cuda/gpus)、[旧 GPU 架构表](https://developer.nvidia.com/cuda/gpus/legacy)、[CUDA / 驱动 / 架构兼容矩阵](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)、[PyTorch 2.9.1 发布兼容说明](https://github.com/pytorch/pytorch/blob/v2.9.1/RELEASE.md)。

## 安装并复用官方底模

```bash
git clone https://github.com/danielsunrise/breeze-tts2-finetuning-v100.git
cd breeze-tts2-finetuning-v100
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

使用仍包含 sm_70 支持的构建；上述 CUDA 12.6 组合来自原验证环境。完整环境版本参考 [requirements.v100.lock.txt](requirements.v100.lock.txt)。

```bash
cp .env.example .env
# 编辑 .env：把 BREEZE_MODEL_ROOT 指向已经下载的完整官方底模。
source .env
```

底模需要官方检查点、tokenizer 与 `audio_tokenizer/`；获取方式见推理仓库。训练输出和缓存必须与底模分开。不要把模型下载到 Git 中，也不要把输出目录指向共享底模。

## GPT-SoVITS 数据转换

可复用 `.list` 中的 `音频路径|说话人|语言|准确文本` 和原始音频。旧 Windows/相对路径按音频文件名在本机目录中唯一映射；不会移动、重采样、改写音频或修改原标注。

```bash
python -m tools.convert_gptsovits \
  --list /path/to/annotations.list \
  --audio-root /path/to/audio \
  --output-root datasets/custom
```

多说话人需加 `--speaker` 明确选择；默认语言 `zh,en`，可用 `--languages` 调整。工具校验音频可读、有限值、非静音，去除重复 PCM；相同规范化文本留在同一个 split，减少短句泄漏。输出包含 `train.jsonl`、`validation.jsonl`、映射及检查报告。**检查不等于人工核对文字、音质或音色。**

也可自行准备独立的 JSONL，每行一个绝对路径和准确文本：

```json
{"audio":"/path/to/audio/clip.wav","text":"音频中完整、准确的说话内容。"}
```

在 `.env` 中把训练与验证路径指向相应清单。原始音频、标注、转换映射和监督缓存都可能包含私人信息，已排除在 Git 跟踪之外。

## 准备全部清单的监督缓存

```bash
python -m tools.prepare_cache
```

从 `BREEZE_*` 环境变量读取底模、清单、缓存路径，按清单实际条数编码，检查训练/验证音频不重叠和缓存不覆盖底模。底层仍调用社区的 `training.real_data`，生成 16 码本监督目标。更多原生参数见 `python -m training.real_data --help`。

## 先跑一个有预算的 LoRA 试训

```bash
python -m tools.bounded_lora --plan configs/v100-lora.example.json
```

示例为 rank 8、alpha 16、最多 200 个优化器步、梯度累积 4、学习率 `1e-4`、warmup 20。最多呈现 800 次样本，并限制最多一轮；训练集不足 800 条时会拒绝开训，需相应减少 `max_steps` 和暂停阶段，不会默默超过轮数预算。

需要调整预算时，先把示例复制为 `datasets/custom/trial-plan.json`，修改副本，再把 `--plan` 指向副本。`datasets/` 已被忽略，避免把个人实验计划公开，也能保持源码工作树干净。

每 50 步保存并检查固定验证子集，连续两次改善未超过 0.01 则提前停止。四个阶段共用同一个 200 步余弦调度，总步数不会因中断重载而重置。`validation_examples` 自动不超过实际验证条数；这里的固定子集参与选检查点，另留独立评估数据才可作更可靠的泛化评估。

输出 `trial-state.json`、`selected-checkpoint.json` 与各阶段 checkpoint。选择按已保存检查点中验证 loss 最低者；loss 下降不等于音色相似或听感更好。预算驱动入口要求新的输出目录；异常中断后，使用下述原生 CLI 和已有 checkpoint 恢复，不要覆盖旧实验。

## 原生 LoRA 与续训

```bash
bash scripts/run_lora.sh \
  --rank 8 --alpha 16 --max-steps 200 --gradient-accumulation 4 \
  --learning-rate 1e-4 --warmup-steps 20 --save-every 50 \
  --validation-examples 16
```

恢复时保持同一个 `BREEZE_RUN_ROOT` 及原来的运行配置，追加：

```bash
--resume-checkpoint "$BREEZE_RUN_ROOT/checkpoint-step-000050"
```

rank、alpha、学习率、总步数等必须与原 `run-config.json` 一致。不要为了分段训练改变调度 horizon。checkpoint 包含 adapter、优化器、调度器、随机状态、GradScaler 和训练状态；这些文件都不应公开提交。

训练入口要求 Git 工作树干净。自行修改源码后先审查并提交代码；把个人配置放在被忽略的 `.env`，实际数据放 `datasets/`，缓存放 `cache/`，训练输出放 `runs/`。

## 全量 SFT 单步检查

准备好短音频缓存、释放其他 GPU 任务后，可用单独输出目录执行：

```bash
python -m training.real_full_sft \
  --model-root "$BREEZE_MODEL_ROOT" --cache-root "$BREEZE_CACHE_ROOT" \
  --output-root runs/full-sft-check \
  --optimizer adafactor --max-steps 1 --gradient-accumulation 1 \
  --warmup-steps 0 --save-every 1 --validation-examples 1 \
  --learning-rate 1e-4 --save-policy none
```

该例只做计算检查，不导出完整模型，也不表示已验证正式全量训练效果。`--save-policy none` 的结果不能当作可续训 checkpoint。其他全量参数、优化器和研究脚本见社区原文档。

## 可选 GPU 会话锁

```bash
bash tools/gpu_session.sh python -m tools.bounded_lora --plan configs/v100-lora.example.json
```

默认只使用本项目锁，不改变任何系统服务。若明确配置 `BREEZE_TTS_SERVICE` 为某个用户级推理服务，会在它原本运行时临时停止，并在命令退出时恢复；原本关闭的服务保持关闭。锁只协调采用同一锁文件的命令，不替代对其他 GPU 进程的检查。

## 推理、评估与许可证

训练出的适配器可接入 [V100 推理仓库](https://github.com/danielsunrise/breeze-tts-v100) 的 `lora.json`；填写实际 rank、alpha、variant 与 adapter 路径即可，不需要再下载另一份底模。训练仓库自带的推理/评估代码作为社区源码保留，V100 网页使用推理仓库的适配版本。ASR/说话人等可选评估依赖见 `requirements-eval.txt`，不包含在基础验证环境中。

```bash
python -m pytest -q
python -m ruff check training evaluation tools
bash scripts/check_public_tree.sh
```

源码按 Apache 2.0 分发；底模、适配器、衍生模型及输出另受 BreezeBlue 模型许可约束，参见 [MODEL_LICENSE.md](MODEL_LICENSE.md)。原社区研究发布说明保留在 `release/`，不代表本仓库附带这些模型或训练结果。

公开仓库不包含个人音色名称、机器地址、私有路径、数据清单、训练录音、adapter、checkpoint、缓存、日志、QQ 集成或访问凭据。发布时使用新的脱敏提交，不携带本机实验 Git 历史。
