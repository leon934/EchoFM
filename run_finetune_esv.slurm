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

torchrun --nproc_per_node=2 run_finetune.py \
    --batch_size 32 \
    --epochs 400 \
    --accum_iter 2 \
    --model mae_vit_base_patch16 \
    --finetune ./models/%j/checkpoint-00715.pth \
    --metric ESV \
    --blr 0.1 \
    --warmup_epochs 30 \
    --output_dir ./checkpoints_ft/esv \
    --log_dir ./logs \
    --data_path ./EchoNet-Dynamic \
    --num_workers 9 \
    --pin_mem \
    --num_frames 16