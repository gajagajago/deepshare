# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import time
import argparse
import operator

# DeepShare: Init system path to `models/DLRM`
import sys, os
import io
sys.path.append(os.path.join(os.environ['DEEPSHARE_PATH'], 'models/fairscale/benchmarks'))

from utils.deepshare_handler import DeepShareSlurmHandler
from utils.checkpoint import DummyDeepShareJobCheckpointer

from golden_configs.lm_wikitext2 import MOE as MOEConfig
from datasets.wikitext2_data import get_dataloaders as get_wikitext2_dataloaders
from datasets.wikitext2_data import get_synthetic_dataloaders as get_synthetic_wikitext2_dataloaders
from models import transformer_lm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.profiler
import torchtext
from distutils.version import LooseVersion
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD
from torchtext.data.utils import get_tokenizer
if operator.ge(torchtext.__version__, LooseVersion("0.10.0")):
    from torchtext.legacy.vocab import build_vocab_from_iterator
else:
    from torchtext.vocab import build_vocab_from_iterator
from collections import namedtuple

import numpy as np

# import utils

# MPI_PORT = 29500


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger('train')


def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_model_and_optimizer(args, device, benchmark_config, model_config):
    """Return instantiated model and optimizer function."""

    if args.model_name == "lm":
        model = get_lm_model(args, device, model_config)

    lr = benchmark_config["lr"]

    if args.optimizer == 'sgd':
        def make_sgd(params):
            return SGD(params, lr=lr)
        optimizer = make_sgd
    else: 
        def make_adam(params):
            return Adam(params, lr=lr)
        optimizer = make_adam

    return model, optimizer


def get_lm_model(args, device, config):
    """Get language model(based on GPT-2) used for sequence prediction."""

    ninp = config["ninp"]
    nhead = config["nhead"]
    initrange = config["initrange"]
    dropout = config["dropout"]
    vocab_size = config["vocab_size"]
    nhid = config["nhid"]
    # DeepShare: Modify read `model_config` value for keys `ndecoder`
    # ndecoder = config["num_decoder_layers"]
    ndecoder = 10
    is_moe = config.get("is_moe", False)
    num_local_experts = config.get("num_local_experts", 1)

    model = transformer_lm.TransformerLM(
        vocab_size, ninp, nhead, nhid, dropout, initrange, ndecoder, is_moe, num_local_experts
    ).to(device)

    return model


def get_synthetic_dataloaders(args, device, benchmark_config, model_specs):
    """Returns dataloader for synthetic data."""

    if args.model_name == "lm":
        return get_synthetic_wikitext2_dataloaders(args, benchmark_config, model_specs)
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def get_real_datasets(args):
    train_filepath = os.path.join(args.data_path, args.dataset, 'wiki.train.tokens')
    valid_filepath = os.path.join(args.data_path, args.dataset, 'wiki.valid.tokens')
    test_filepath = os.path.join(args.data_path, args.dataset, 'wiki.test.tokens')

    tokenizer = get_tokenizer("basic_english")

    def data_process(raw_text_iter):
        data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    DatasetsInfo = namedtuple("DataSetsInfo", ["ntokens", "train_dataset", "valid_dataset", "test_dataset"])

    vocab = build_vocab_from_iterator(map(tokenizer, iter(io.open(train_filepath, encoding="utf8"))))

    train_dataset = data_process(iter(io.open(train_filepath, encoding="utf8")))
    valid_dataset = data_process(iter(io.open(valid_filepath, encoding="utf8")))
    test_dataset = data_process(iter(io.open(test_filepath, encoding="utf8")))

    return DatasetsInfo(len(vocab.stoi), train_dataset, valid_dataset, test_dataset)


def get_real_dataloaders(args, device, benchmark_config, model_specs):
    """Returns dataloaders for real data."""

    if args.model_name == "lm":
        datasets_info = get_real_datasets(args)
  
        train_dataloader, valid_dataloader, test_dataloader = get_wikitext2_dataloaders(datasets_info, benchmark_config, model_specs, 
                                                                                        num_replicas=args.world_size, rank=args.rank)

        model_specs["vocab_size"] = datasets_info.ntokens

        return train_dataloader, valid_dataloader, test_dataloader
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def create_model_config(args, benchmark_config=None, model_specs=None):
    """Return a dict with the given model, dataset and optimizer."""

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.use_synthetic_data:
        dataloader_fn = get_synthetic_dataloaders
    else:
        dataloader_fn = get_real_dataloaders

    data = dataloader_fn(args, device, benchmark_config, model_specs)
    model, optimizer = get_model_and_optimizer(args, device, benchmark_config, model_specs)
    return {
        "model": model,
        "optimizer": optimizer,
        "data": data,
    }


def create_benchmark_config(model_name, config_class):
    """Return a dict with configurations required for benchmarking `model_name` model."""

    if model_name == "lm":
        return config_class.get_benchmark_config()
    else:
        raise RuntimeError("Unrecognized model_mame " % model_name)


def get_model_specs(model_name, config_class):
    """Return a dict with configurations required for configuring `model_name` model."""

    if model_name == "lm":
        return config_class.get_model_config()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def benchmark_moe(config_class, args, slurm_handler):
    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] benchmark_moe')

    os.environ['MASTER_ADDR'] = f'{slurm_handler.master_addr}'
    os.environ['MASTER_PORT'] = f'{slurm_handler.master_port}'
    
    torch.distributed.init_process_group(backend="nccl", rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['WORLD_SIZE']))
    torch.cuda.set_device(0)
    init_random_seed(0)

    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['SLURM_PROCID'])
    
    _logger.debug(
        f"master address = {slurm_handler.master_addr}, "
        + f"master port = {slurm_handler.master_port}, "
        + f"[{os.getpid()}]: world_size = {torch.distributed.get_world_size()}, "
        + f"rank = {torch.distributed.get_rank()}, "
        + f"visible devices={os.environ['CUDA_VISIBLE_DEVICES']} \n"
    )

    # DeepShare: Modify read `benchmark_config` value for keys `epochs` and `batch_size`
    benchmark_config = create_benchmark_config(args.model_name, config_class)
    benchmark_config['epochs'] = args.epochs
    benchmark_config['batch_size'] = args.batch_size

    model_specs = get_model_specs(args.model_name, config_class)
    model_config = create_model_config(args, benchmark_config=benchmark_config, model_specs=model_specs)
    
    model = model_config["model"]

    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] benchmark_config {benchmark_config}')
    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] model_specs {model_specs}')
    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] model_config {model_config}')

    moe_model = DDP(model, device_ids=[0], output_device=0, broadcast_buffers=False)
    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Created MoE model')

    benchmark_language_model(model_config, moe_model, benchmark_config, model_specs, args, slurm_handler)


def train(model_config, model, benchmark_config, model_specs, args, slurm_handler):
    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] train')

    lm_dataloader, _, _ = model_config["data"]
    criterion = benchmark_config["criterion"]
    vocab_size = model_specs["vocab_size"]

    model.train()

    optimizer = model_config["optimizer"]
    optimizer = optimizer(model.parameters())
    group = model.group if hasattr(model, "group") else None

    def get_batch(source):
        seq_len = len(source) - 1
        data = source[0:seq_len]
        target = source[1 : 1 + seq_len]
        return data, target

    pass

    total_loss = 0.0
    word_counter = 0
    total_tokens = 0
    total_tokens_per_log_interval = 0
    bptt = 2

    total_elapsed = 0.0

    # lm_dataloader, _, _ = utils.get_data_loader(
    #     model_config["dataset_info"], args, benchmark_config, model_specs, num_replicas=world_size, rank=rank
    # )

    for i, batch in enumerate(lm_dataloader):

        if i == 1:
            epoch_start_time = time.time()

        batch_start_time = time.time()

        source, target = get_batch(batch)
        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Get batch')

        if i > 0:
            total_tokens += source.numel()

        optimizer.zero_grad()
        input = source.cuda()
        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Input copied to device')
        target = target.cuda()
        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Target copied to device')
        output = model(input)
        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Output computed')  

        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Gradient computed')      

        torch.nn.utils.clip_grad_value_(model.parameters(), model_specs["clip_value"])
        optimizer.step()
        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Optimizer step')  

        total_loss += loss.item()

        ### Notify DeepShare profiler about iteration training status
        batch_time = time.time() - batch_start_time
        if slurm_handler.profiler != None:
            slurm_handler.profiler.step(samples=args.batch_size * args.world_size, bt=batch_time)

    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Before torch.cuda.sync')  
    if epoch_start_time != 0:
        torch.cuda.synchronize()
        wps = total_tokens / (time.time() - epoch_start_time)
    else:
        raise RuntimeError(
            "Unable to benchmark on a single batch. Increase the size " " of the dataset and rerun the benchmark."
        )
    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] After torch.cuda.sync')  

    return wps, loss.item()


# def benchmark_language_model(rank, world_size, benchmark_config, model_specs, args):
def benchmark_language_model(model_config, model, benchmark_config, model_specs, args, slurm_handler):
    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] benchmark_language_model')

    epochs = benchmark_config["epochs"]
    if dist.get_rank() == 0:
        _logger.info(f'Total epochs: {epochs}')

    # Wait for another job
    if args.wait_workers != 0:
        slurm_handler.file_log(args.polling_file_path, 'r', mode='a')
        slurm_handler.file_poll(args.polling_file_path, 'r' * args.wait_workers)
        slurm_handler.print_with_rank('polling done!')
    
    ### Install DeepShare profiler
    if args.profile_gpu:
        slurm_handler.profiler.start_gpu_profile()
    if args.profile_cpu:
        slurm_handler.profiler.start_cpu_profile()
    if args.profile_progress:
        pass
    
    for epoch in range(epochs):
        start_time = time.time()
        if dist.get_rank() == 0:
            _logger.debug("-" * 110)
            _logger.debug("| start of epoch {:1d}".format(epoch))
            _logger.debug("-" * 110)
        wps, loss = train(model_config, model, benchmark_config, model_specs, args, slurm_handler)
        elapsed_time = time.time() - start_time
        if dist.get_rank() == 0:
            _logger.debug("-" * 110)
            _logger.debug("| end of epoch {:1d} | time: {:5.2f}s | train loss {:5.2f} ".format(epoch, elapsed_time, loss))
            _logger.debug("-" * 110)
            _logger.debug("Throughput(wps) is {:.2f}.".format(wps))


parser = argparse.ArgumentParser(description="benchmark")
parser.add_argument("--chunks", type=int, default=1, help="number of microbatches per batch")
parser.add_argument("--use_synthetic_data", action="store_true", help="Uses synthetic data for running benchmarks.")
parser.add_argument(
    "--model_name",
    default="lm",
    help="Language Model(LM) used to benchmark FSDP.",
)

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
parser.add_argument('--data-path', default='/cmsdata/ssd0/cmslab/dlcm_data/Fairscale-data', type=str,
                    help='path to dataset')                
parser.add_argument('--dataset', default='wikitext-2', type=str,
                    help='Train dataset, one of (wikitext-2, wikitext-103). (default: wikitext-2)')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='Optimizer, one of (adam, sgd). (default: sgd)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
                    help='Input batch size for training (default: 8)')
parser.add_argument("--wait-workers", type=int, default=0,
                    help='how many workers to wait for getting prepared(including itself)')
parser.add_argument('--polling-file-path', default='./out/ready', type=str, metavar='PATH',
                    help='path to polling file')

if __name__ == "__main__":
    args = parser.parse_args()

    # DeepShare: Setup log level
    if args.debug:
        _logger.info(f'Set log level to {logging.DEBUG}')
        _logger.setLevel(logging.DEBUG)

    _logger.info(f"Running MoE benchmark with args: {args}")

    slurm_handler = DeepShareSlurmHandler(DummyDeepShareJobCheckpointer()) # TODO: add checkpointer
    if args.profile_gpu or args.profile_cpu or args.profile_progress:
        slurm_handler.install_profiler(profile_path=args.profile_path, profile_gpu=args.profile_gpu, 
                                        profile_cpu=args.profile_cpu, profile_progress=args.profile_progress, 
                                        profile_iteration=args.profile_iteration)

    # Train entry
    benchmark_moe(MOEConfig, args, slurm_handler)