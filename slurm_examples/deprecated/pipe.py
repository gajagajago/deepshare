# Copyright 2021 TUNiB Inc.
import argparse
import numpy as np
import random
import time
from datasets import load_from_disk

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader

from transformers import AutoTokenizer

# DeepShare: Additional required packages
import logging
import os
import sys

# DeepShare: Init system path to `models/DeepSpeedPipelineParallel`
sys.path.append(os.path.join(os.environ['DEEPSHARE_PATH'], 'models/DeepSpeedPipelineParallel'))
from model.GPT2Pipe import GPT2ForSequenceClassificationPipe
from data import ClassificationDataset, collate_fn

from utils.deepshare_handler import DeepShareSlurmHandler
from utils.checkpoint import DummyDeepShareJobCheckpointer

# DeepShare: Initialized global `_logger` and `args`.
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger('train')
args = None


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


class TestPPTraining:
    def __init__(self, args):
        self.args = args

        seed_everything(args.seed)
        self.total_step = args.total_step
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token


    def test_pp_for_sequence_classification(self, num_labels, model_name, pp_stage):

        ### Create model
        model_ds = GPT2ForSequenceClassificationPipe.from_pretrained(
            model_name, num_labels=num_labels)

        model_ds.config.pad_token_id = self.tokenizer.eos_token_id

        net = PipelineModule(layers=model_ds.to_layers(),
                             loss_fn=model_ds.loss_fn, num_stages=pp_stage)

        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Model created')

        ### Create optimizer
        optimizer_ds = Adam(net.parameters(), lr=1e-5, weight_decay=1e-5)

        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Optimizer created')

        ### Entrypoint for training w/ DeepSpeed
        # TODO: Fix error for multi-node training
        # 
        # Currently, multi-node training hangs at p2p.init_process_groups (https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/engine.py)
        # or (https://github.com/pytorch/pytorch/blob/master/torch/distributed/distributed_c10d.py)
        # 
        # Error log)
        # Waiting in store based barrier to initialize process group for rank: 0/1, key: 
        # store_based_barrier_key:13 (world_size=4, worker_count=2, timeout=0:30:00) 
        # 
        # Seems that processes on the second node is not getting here
        engine, _, _, _ = deepspeed.initialize(
            args=args,
            model=net,
            model_parameters=[p for p in net.parameters() if p.requires_grad],
            optimizer=optimizer_ds)

        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] deepspeed.initialize')

        ### Load dataset
        dataset = load_from_disk(self.args.data_path).data["train"]

        # truncation because tokenizing each data took too long
        premise, hypothesis, label = dataset[2][:200], dataset[5][:200], dataset[9][:200]
    
        dataset = [(p, h, l) for p, h, l in zip(premise, hypothesis, label)]
        dataset = ClassificationDataset(dataset, self.tokenizer)

        dataloader = DataLoader(
            dataset,
            batch_size=engine._config.train_batch_size,
            num_workers=int(os.environ['SLURM_CPUS_PER_TASK']), # DeepShare: May cause error if `--cpus-per-task` for sbatch script is not set
            drop_last=True,
            collate_fn=collate_fn,
            sampler=DistributedSampler(
                dataset,
                num_replicas=engine.grid.get_data_parallel_world_size(),
                rank=engine.grid.get_data_parallel_rank(),
                shuffle=True))

        _logger.info(f'[Rank {os.environ["SLURM_PROCID"]}] len(dataloader): {len(dataloader)}')

        dataloader = RepeatingLoader(dataloader)
        train_iter = iter(dataloader)

        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Dataloader')

        ### Install DeepShare profiler
        if args.profile_gpu:
            slurm_handler.profiler.start_gpu_profile()
        if args.profile_cpu:
            slurm_handler.profiler.start_cpu_profile()
        if args.profile_progress:
            pass

        """ Epoch entry point """
        for epoch in range(args.epochs):

            """ Step entry point """
            for i, _ in enumerate(iter(dataloader)):

                batch_start_time = time.time()
                
                engine.train_batch(train_iter)

                ### Notify DeepShare profiler about iteration training status
                batch_time = time.time() - batch_start_time
                if slurm_handler.profiler != None:
                    slurm_handler.profiler.step(samples=engine._config.train_batch_size, bt=batch_time)


def main(slurm_handler):

    os.environ['MASTER_ADDR'] = f'{slurm_handler.master_addr}'
    os.environ['MASTER_PORT'] = f'{slurm_handler.master_port}'
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    os.environ['LOCAL_RANK'] = '0'

    deepspeed.init_distributed(dist_backend=args.backend)
    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] init_distributed')

    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['SLURM_PROCID'])

    _logger.debug(
        f"master address = {slurm_handler.master_addr}, "
        + f"master port = {slurm_handler.master_port}, "
        + f"[{os.getpid()}]: world_size = {torch.distributed.get_world_size()}, "
        + f"rank = {torch.distributed.get_rank()}, "
        + f"visibile devices={os.environ['CUDA_VISIBLE_DEVICES']} \n")

    test = TestPPTraining(args)

    test.test_pp_for_sequence_classification(
        model_name=args.model_name, num_labels=3, pp_stage=args.pp_stage)


def get_args():
    parser = argparse.ArgumentParser(description='GPT2')
    parser.add_argument("--model_name", type=str,
                        default='gpt2')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--total_step',
                        type=int,
                        default=10,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pp-stage',
                        type=int,
                        default=1,
                        help='pipeline parallelism. Must be a dividend of WORLD_SIZE')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')

    # Custom argument added for DeepShare
    parser.add_argument('--debug', action='store_true', default=False,
                    help='set log level to loggig.debug')
    parser.add_argument('--hdfs-ckpt-dir', default='', type=str, metavar='PATH',
                        help='HDFS directory to store checkpoint (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--accumulate-iteration', type=int, default=1,
                        help='batch interval to accumulate gradient without synchronization')
    parser.add_argument('--profile-gpu', action='store_true', default=False,
                        help='profile with torch profiler')
    parser.add_argument('--profile-cpu', action='store_true', default=False,
                        help='profile CPU util')
    parser.add_argument('--profile-progress', action='store_true', default=False,
                        help='profile progres')
    parser.add_argument('--profile-iteration', type=int, default=3,
                        help='batch indices to profile with torch profiler')
    parser.add_argument('--profile-path', default=f'./out/log/{os.environ["SLURM_JOBID"]}', type=str,
                        help='path to write profiler logs to.')
    parser.add_argument('--data-path', default='/cmsdata/ssd0/cmslab/dlcm_data/deepspeed-pp-data', type=str,
                        help='path to dataset')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser = deepspeed.add_config_arguments(parser) # Read `--deepspeed_config` arguments (including batch size) and add it to `args`. 

    args = parser.parse_args()

    # args constraints
    assert int(os.environ['WORLD_SIZE']) % args.pp_stage == 0 

    return args


if __name__ == "__main__":

    args = get_args()

    # DeepShare: Setup log level
    if args.debug:
        _logger.info(f'Set log level to {logging.DEBUG}')
        _logger.setLevel(logging.DEBUG)

    _logger.info(f"Running Pipe benchmark with args: {args}")

    slurm_handler = DeepShareSlurmHandler(DummyDeepShareJobCheckpointer()) # TODO: add checkpointer
    if args.profile_gpu or args.profile_cpu or args.profile_progress:
        slurm_handler.install_profiler(profile_path=args.profile_path, profile_gpu=args.profile_gpu, 
                                        profile_cpu=args.profile_cpu, profile_progress=args.profile_progress, 
                                        profile_iteration=args.profile_iteration)
    
    # Train entry
    main(slurm_handler)