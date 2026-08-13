# TAPrompt: Transfer-Aware Prompt for Continual Learning via Topology-Guided Memory Synthesis

TAPrompt is a PyTorch implementation for prompt-based class-incremental learning with Vision Transformers. This repository is prepared for review and reproducibility. 

## Environment

The code was developed with Python 3.9.

```bash
conda create -n taprompt python=3.9
conda activate taprompt
pip install -r requirements.txt
```

Install a PyTorch / torchvision build that matches your CUDA version. The original baseline used PyTorch 1.11 and torchvision 0.12; newer versions may also work but should be checked carefully.

## Data

Set `--data-path` to your local dataset directory. You can either update this argument in the scripts under `training_scripts/` or pass it manually when running `main.py`.

Expected datasets:

- CIFAR-100
- ImageNet-R
- CUB-200-2011
- RESISC45

For `Split-RESISC45-transfer`, the code filters RESISC45 into five predefined tasks and remaps labels to a compact 15-class label space.

## Running Experiments

Run commands from the repository root.

```bash
cd ./TAPrompt
```

Example: Split-CIFAR100 with supervised ViT.

```bash
bash training_scripts/train_cifar100_sup.sh
```

Example: Split-CUB200 with DINO.

```bash
bash training_scripts/train_cub_dino.sh
```

Example: Split-RESISC45-transfer.

```bash
bash training_scripts/train_resisc45_sup_tranfer.sh
```

The scripts are meant as reproducibility templates. Please check `CUDA_VISIBLE_DEVICES`, `--master_port`, `--data-path`, `--output_dir`, and `--trained_taprompt_model` before running them.

## Evaluation

Evaluation is enabled by adding `--eval` and setting `--trained_taprompt_model` to an output directory containing task checkpoints:

```text
<trained_taprompt_model>/checkpoint/task1_checkpoint.pth
<trained_taprompt_model>/checkpoint/task2_checkpoint.pth
...
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29411 \
  --use_env main.py \
  cifar100_taprompt \
  --model vit_base_patch16_224 \
  --batch-size 24 \
  --epochs 50 \
  --data-path /path/to/local_datasets/ \
  --output_dir ./output/cifar100_eval \
  --trained_taprompt_model ./output/cifar100_train \
  --num_tasks 10 \
  --size 10 \
  --eval
```

Evaluation logs are written to `output_dir` as `test_stats_<timestamp>_<id>.txt`.

## Acknowledgement

This repository is built on CAPrompt and HiDe-Prompt.

## Citation

If you find this code useful, please cite our paper. The BibTeX entry will be updated after publication.
