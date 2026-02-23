#!/bin/bash
#SBATCH -J echofm_finetune_ef
#SBATCH -A 132677318890
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o finetune_ef_%j.out

echo "Starting job on $(hostname)"

module load GCCcore/12.2.0 Python/3.10.8
source $SCRATCH/echofm-env/bin/activate
cd ~/EchoFM

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12356
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

torchrun \
  --nproc_per_node=2 \
  run_finetune.py \
  --data_path $SCRATCH/EchoNet-Dynamic \
  --batch_size 8 \
  --epochs 100 \
  --accum_iter 2 \
  --model mae_vit_large_patch16 \
  --finetune $SCRATCH/EchoFM_pretrain/checkpoint-00299.pth \
  --metric EF \
  --blr 0.001 \
  --warmup_epochs 5 \
  --output_dir $SCRATCH/EchoFM_finetune_EF \
  --log_dir $SCRATCH/EchoFM_finetune_EF_logs \
  --num_workers 6 \
  --num_frames 32