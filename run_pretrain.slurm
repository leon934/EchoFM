#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --output=./jobs/%j_%x_output
#SBATCH --nodes=1   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu

#First Executable Line
source ~/.bashrc
module load GCCcore/12.2.0 Python/3.10.8
source .venv/bin/activate

torchrun --nproc_per_node=2 run_pretrain.py \
    --data_path ./EchoNet-Dynamic \
    --batch_size 32 \
    --log_dir ./logs \
    --output_dir ./models/%j \
    --model mae_vit_base_patch16 \
    --blr 0.0001 \
    --warmup_epochs 40 \
    --epochs 800 \
    --mask_ratio 0.75 \
    --pin_mem \
    --num_workers 4 \
    --accum_iter 2 \
    --weight_decay 0.05 \
    --norm_pix_loss \
    --num_frames 16