# Night Enhancement

一个基于 PyTorch 的低照度图像增强项目，支持 LOL / LOLv2 数据集训练、评估与推理。

## 功能特性

- 支持 LOL 与 LOLv2 数据集
- 提供自动数据下载与数据划分脚本
- 支持训练、评估、推理、速度基准测试
- 训练过程自动保存 best 与周期性 checkpoint
- 使用 JSONL 记录训练日志，便于后处理分析

## 项目结构

```text
configs/        # 数据、模型、训练配置
datasets/       # 数据集读取逻辑
evaluators/     # 评价指标
losses/         # 损失函数
models/         # 模型定义
scripts/        # 训练/评估/推理/数据准备脚本
trainers/       # 训练器
utils/          # 工具函数
data/           # 数据目录（已在 .gitignore 中忽略）
outputs/        # 权重与日志输出（已在 .gitignore 中忽略）
```

## 环境安装

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

依赖包含：

- torch >= 2.2.0
- torchvision >= 0.17.0
- numpy
- Pillow
- PyYAML
- tqdm
- scikit-image

## 数据准备

### 方式 1：自动下载并整理（推荐）

```bash
python scripts/download_datasets.py --datasets lol
python scripts/download_datasets.py --datasets lolv2
```

同时准备两个数据集：

```bash
python scripts/download_datasets.py --datasets lol lolv2
```

### 方式 2：使用本地已下载压缩包

把 `LOL.zip` / `LOLv2.zip` 放到某个目录后执行：

```bash
python scripts/download_datasets.py --datasets lol lolv2 --local_archive_dir data/manual_archives
```

### 方式 3：手动准备后生成 split

若你已经有 `low/` 与 `high/` 目录，可手动生成 LOL 的 split 文件：

```bash
python scripts/prepare_splits.py --root data/LOL --low_dir low --high_dir high --out_dir data/splits
```

## 配置文件说明

- `configs/dataset.yaml`：LOL 默认数据配置
- `configs/dataset_lolv2.yaml`：LOLv2 默认数据配置
- `configs/model.yaml`：模型结构超参数
- `configs/train.yaml`：训练参数（epoch、batch size、lr、loss 权重等）

可按需修改其中参数后再运行脚本。

## 训练

默认用 LOL 配置训练：

```bash
python scripts/train.py \
  --dataset_cfg configs/dataset.yaml \
  --model_cfg configs/model.yaml \
  --train_cfg configs/train.yaml
```

使用 LOLv2 训练：

```bash
python scripts/train.py \
  --dataset_cfg configs/dataset_lolv2.yaml \
  --model_cfg configs/model.yaml \
  --train_cfg configs/train.yaml
```

训练输出默认在 `outputs/`：

- `best.pt`
- `epoch_*.pt`
- `train_log.jsonl`

## 评估

```bash
python scripts/evaluate.py \
  --dataset_cfg configs/dataset.yaml \
  --model_cfg configs/model.yaml \
  --checkpoint outputs/best.pt
```

输出指标：

- MSE
- PSNR
- SSIM
- IE

## 推理

```bash
python scripts/infer.py \
  --model_cfg configs/model.yaml \
  --checkpoint outputs/best.pt \
  --input_dir data/LOL/low \
  --output_dir outputs/infer \
  --image_size 256
```

## 速度基准测试

```bash
python scripts/benchmark.py --model_cfg configs/model.yaml --size 256 --warmup 20 --runs 100
```

## 消融实验（当前脚本）

```bash
python scripts/ablation.py --dry_run
```

说明：当前 `ablation.py` 会打印每个设置的说明，但实际仍调用同一训练命令，若要真正应用 override，需要在脚本中补充参数传递逻辑。

## GitHub 提交建议

仓库已建议忽略以下内容（见 `.gitignore`）：

- `data/`
- `outputs/`
- `__pycache__/`
- `*.pt` / `*.pth`

提交代码时建议只提交源码和配置，不提交数据集与模型权重。

## 常见问题

1. checkpoint 找不到
- 检查 `--checkpoint` 路径，或确认 `outputs/best.pt` 是否存在。

2. 显存不足
- 降低 `configs/train.yaml` 中的 `batch_size`。

3. 数据集配对失败
- 检查目录命名是否符合 low/high（或 low/normal）语义；必要时使用 `--local_archive_dir` 或手动生成 split。
