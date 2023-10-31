# Copyright 2021 TUNiB Inc.
import argparse
import copy

import deepspeed
import torch
import torch.distributed as dist
from datasets import load_dataset

from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)
import numpy as np
import random


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


class TestPPTraining:
    def __init__(self, args):
        # Setting random
        seed_everything(42)

        self.args = args
        self.total_step = args.total_step
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_pp_for_sequence_classification(self, num_labels, model_name, pp_stage):

        model_ds = GPT2ForSequenceClassificationPipe.from_pretrained(
            model_name, num_labels=num_labels)

        model_ds.config.pad_token_id = self.tokenizer.eos_token_id

        deepspeed.init_distributed()
        net = PipelineModule(layers=model_ds.to_layers(),
                             loss_fn=model_ds.loss_fn, num_stages=pp_stage)

        optimizer_ds = Adam(net.parameters(), lr=1e-5, weight_decay=1e-5)

        # dataset (should be tuple of tensor)
        dataset = load_dataset("multi_nli").data["train"]

        # truncation because tokenizing each data took too long
        premise, hypothesis, label = dataset[2][:
                                                200], dataset[5][:200], dataset[9][:200]

        dataset = [(p, h, l) for p, h, l in zip(premise, hypothesis, label)]
        dataset = ClassificationDataset(dataset, self.tokenizer)

        engine, _, _, _ = deepspeed.initialize(
            args=args,
            model=net,
            model_parameters=[p for p in net.parameters() if p.requires_grad],
            optimizer=optimizer_ds)

        dataloader = DataLoader(
        dataset,
        batch_size=engine._config.train_batch_size,
        num_workers=24,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=DistributedSampler(
            dataset,
            num_replicas=engine.grid.get_data_parallel_world_size(),
            rank=engine.grid.get_data_parallel_rank(),
            shuffle=True,
            ),
        )

        dataloader = RepeatingLoader(dataloader)
        train_iter = iter(dataloader)

        for _ in range(100):
            engine.train_batch(train_iter)


def main(args):
    deepspeed.init_distributed(dist_backend=args.backend)

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
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')

    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        print(path.dirname(path.dirname(path.abspath(__file__))))
        sys.path.append((path.dirname(path.dirname(path.abspath(__file__)))))
        from model.GPT2Pipe import GPT2ModelPipe, GPT2ForSequenceClassificationPipe
        from data import ClassificationDataset, collate_fn
    else:
        from ..model.GPT2Pipe import GPT2ModelPipe, GPT2ForSequenceClassificationPipe
        from ..data import ClassificationDataset, collate_fn

    args = get_args()
    main(args)
