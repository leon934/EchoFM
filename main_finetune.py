# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time

# import mae_st.util.env

import mae_st.util.misc as misc

import numpy as np
# import timm
import torch
import torch.backends.cudnn as cudnn
from iopath.common.file_io import g_pathmgr as pathmgr
from mae_st import models_mae
from mae_st.engine_finetune import train_one_epoch, evaluate
from mae_st.util.misc import NativeScalerWithGradNormCount as NativeScaler

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data.dataset import EchoDataset_from_Video_mp4
import torch.distributed as dist


def get_args_parser():
    parser = argparse.ArgumentParser("MAE Linear Probing / Fine-tuning", add_help=False)
    
    # Training Config
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    
    # Model Config
    parser.add_argument("--model", default="mae_vit_large_patch16", type=str, metavar="MODEL")
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--finetune", default="", help="Path to checkpoint for linear probing")
    parser.add_argument("--metric", choices=["EF", "ESV", "EDV"], help="metric type to finetune")
    
    # Optimizer
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR", help="base learning rate")
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=5, metavar="N")

    # Data / Paths
    parser.add_argument("--output_dir", default="./output_dir_linprobe")
    parser.add_argument("--log_dir", default="./output_dir_linprobe")
    parser.add_argument("--data_path", default="/raid/camca/sk1064/us/fullset/video/", help="Training data path")
    
    # Environment
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from training checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_env", action="store_true")
    parser.set_defaults(pin_mem=True)
    
    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--distributed", default=True, type=bool)

    # MAE Specifics (Must be present to initialize model, even if unused for probing)
    parser.add_argument("--mask_ratio", default=0.0, type=float) # 0 for probing
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.set_defaults(norm_pix_loss=False)
    
    # Video Specifics
    parser.add_argument("--decoder_embed_dim", default=512, type=int)
    parser.add_argument("--decoder_depth", default=8, type=int)
    parser.add_argument("--decoder_num_heads", default=16, type=int)
    parser.add_argument("--t_patch_size", default=4, type=int)
    parser.add_argument("--num_frames", default=32, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--sampling_rate", default=4, type=int)
    parser.add_argument("--repeat_aug", default=1, type=int) # Usually 1 for fine-tuning/probing
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--num_checkpoint_del", default=20, type=int)
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument("--trunc_init", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.set_defaults(fp32=True)
    parser.add_argument("--jitter_scales_relative", default=[0.5, 1.0], type=float, nargs="+")
    parser.add_argument("--jitter_aspect_relative", default=[0.75, 1.3333], type=float, nargs="+")
    parser.add_argument("--beta", default=None, type=float, nargs="+")
    parser.add_argument("--pred_t_dim", type=int, default=8)
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)

    # Misc
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--cpu_mix", action="store_true", help="Perform mixup on CPU")

    return parser

class LinearProbeMAE(nn.Module):
    """
    Wraps a pre-trained MAE model for linear probing.
    Freezes the backbone, applies BN (affine=False) to the CLS token,
    and adds a Linear classification head.
    """
    def __init__(self, mae_model, num_classes):
        super().__init__()
        self.mae = mae_model

        decoder_attrs = [
            'decoder_blocks', 
            'decoder_embed', 
            'decoder_pred', 
            'decoder_norm',
            'decoder_pos_embed', 
            'decoder_pos_embed_spatial', 
            'decoder_pos_embed_temporal', 
            'decoder_pos_embed_class',
            'decoder_cls_token', 
            'mask_token' # Only used in decoder for reconstruction
        ]
        
        for attr in decoder_attrs:
            if hasattr(self.mae, attr):
                delattr(self.mae, attr)

        # Un-Freeze MAE parameters
        #for param in self.mae.parameters():
        #    param.requires_grad = False
            
        # Get embedding dimension (Standard MAE/ViT has embed_dim attribute)
        embed_dim = mae_model.norm.weight.shape[0] 
        
        # Batch Norm without affine as per MAE paper for linear probing
        #self.bn = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)   

        # Initialize head
        nn.init.constant_(self.head.bias, 0)
        nn.init.normal_(self.head.weight, std=0.01)

    def forward(self, x):
        # MAE encoder forward with mask_ratio=0
        # Returns: (latent, mask, ids_restore)
        latent, _, _ = self.mae.forward_encoder(x, mask_ratio=0.0)
        
        # Extract CLS token (index 0)
        cls_token = latent[:, 0]
        
        # Apply normalization and head
        #x = self.bn(cls_token)
        x = self.head(x)
        return x

def main(args):
    args.distributed = False
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    # 멀티 GPU 초기화
    # if args.distributed:
    #     dist.init_process_group(backend="nccl", init_method=args.dist_url, rank=args.local_rank, world_size=args.world_size)
    #     torch.cuda.set_device(args.local_rank)
    if args.distributed:
        if not dist.is_initialized():  # 이미 초기화된 경우 중복 호출 방지
            dist.init_process_group(
                backend="nccl", 
                init_method=args.dist_url, 
                rank=args.local_rank, 
                world_size=args.world_size
            )
        torch.cuda.set_device(args.local_rank)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = EchoDataset_from_Video_mp4(
        folder=args.data_path,
        num_frames=args.num_frames,
        target_metric=args.metric
    )

    dataset_val = EchoDataset_from_Video_mp4(
        folder=args.data_path,
        num_frames=args.num_frames,
        split="VAL",
        target_metric=args.metric
    )
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True,
        timeout=300,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True,
        timeout=300,
    )

    # define the model
    model = models_mae.__dict__[args.model](
        **vars(args),
    )

    checkpoint = torch.load(args.finetune, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint["model"]

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Missing keys:\n{missing_keys}")
    print(f"Unexpec keys:\n{unexpected_keys}")

    model = LinearProbeMAE(model, num_classes=1)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            # find_unused_parameters=True,
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(
        model_without_ddp,
        args.weight_decay,
        bias_wd=args.bias_wd,
    )
    if args.beta is None:
        beta = (0.9, 0.95)
    else:
        beta = args.beta
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=beta,
    )
    loss_scaler = NativeScaler(fp32=args.fp32)
    criterion = torch.nn.MSELoss()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            fp32=args.fp32,
        )
        if args.output_dir and (
            epoch % args.checkpoint_period == 0 or epoch + 1 == args.epochs
        ):
            checkpoint_path = misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        val_stats = evaluate(data_loader_val, model, device)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with pathmgr.open(
                f"{args.output_dir}/log.txt",
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]


def launch_one_thread(
    local_rank,
    shard_rank,
    num_gpus_per_node,
    num_shards,
    init_method,
    output_path,
    opts,
    stats_queue,
):
    print(opts)
    args = get_args_parser()
    args = args.parse_args(opts)
    args.rank = shard_rank * num_gpus_per_node + local_rank
    args.world_size = num_shards * num_gpus_per_node
    args.gpu = local_rank
    args.dist_url = init_method
    args.output_dir = output_path
    output = main(args)
    stats_queue.put(output)