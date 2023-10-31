# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

""" Modifications
This script was modified from Fairscale FSDP benchmark 
(https://github.com/facebookresearch/fairscale/blob/main/benchmarks/fsdp.py)
Modifications have been made to the training script, with assuming Slurm to launch as sbatch job. Such assumption can
be observed from usage of SLURM_xx environment variables. 
"""

""" Modifications for DeepShare 
- Train dataset can be either WikiText-2/WikiText-103. WikiText-103 is x45 larger than the former, requiring larger device memory and communication cost. (https://huggingface.co/datasets/wikitext) 
- Optimizer can be either Adam/SGD. SGD requires less device memory for optimizer states.
- FP16 training is configured to be default True to avoid device OOM, especially on `loss.backward`.
"""

import argparse
from collections import defaultdict
from functools import reduce
import gc
import logging
import math
import operator
import time

# DeepShare: Init system path to `models/DLRM`
import sys, os
import io
sys.path.append(os.path.join(os.environ['DEEPSHARE_PATH'], 'models/fairscale/benchmarks'))

from utils.deepshare_handler import DeepShareSlurmHandler
from utils.checkpoint import DummyDeepShareJobCheckpointer

import torch.profiler
import torchtext
from distutils.version import LooseVersion
from torchtext.data.utils import get_tokenizer
if operator.ge(torchtext.__version__, LooseVersion("0.10.0")):
    from torchtext.legacy.vocab import build_vocab_from_iterator
else:
    from torchtext.vocab import build_vocab_from_iterator
from collections import namedtuple

from datasets.wikitext2_data import get_dataloaders as get_wikitext2_dataloaders
from datasets.wikitext2_data import get_synthetic_dataloaders as get_synthetic_wikitext2_dataloaders
from models import transformer_lm
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD

from benchmarks.golden_configs.lm_wikitext2 import FSDP as lm_wikitext2
from fairscale.nn import auto_wrap, default_auto_wrap_policy, enable_wrap
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP


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
    ndecoder = config["num_decoder_layers"]

    return transformer_lm.TransformerLM(vocab_size, ninp, nhead, nhid, dropout, initrange, ndecoder).to(device)


def get_tensors_by_size_bucket():

    size_buckets = defaultdict(int)
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue
        if obj.device.type == "cuda":
            size_buckets[(*obj.size(),) + (obj.element_size(),)] += 1

    return size_buckets


def log_number_of_parameters(model):

    num_params = reduce(operator.add, (reduce(operator.mul, x.size()) for x in model.parameters()))
    if hasattr(model, "group"):
        total = torch.Tensor([num_params])
        if torch.cuda.is_available():
            total = total.cuda()
        torch.distributed.all_reduce(total, group=model.group)
        _logger.info(
            f"training model, #params = {num_params/10**6}M, group: {model.group.rank()}, grank:"
            f" {torch.distributed.get_rank()}, sizes {model.group.size()}"
        )
        torch.distributed.barrier()
        if model.group.rank() == 0:
            _logger.info(f"total #prams = {total.item()}")
    else:
        _logger.info(f"training model, #params = {num_params/10**6}M")


def get_device(model, index):
    if isinstance(model, DDP):
        model = model.module

    if not torch.cuda.is_available():
        return torch.device("cpu")
    if hasattr(model, "devices"):
        return model.devices[index]
    else:
        return torch.cuda.current_device()


def get_fake_dataloader(lm_dataloader_len, args):
    fake_input = {"input": torch.zeros(args.batch_size)}

    class FakeDataset:
        def __getitem__(self, index):
            return fake_input

        def __len__(self):
            return lm_dataloader_len

    return FakeDataset()


def train(model_config, model, benchmark_config, model_specs, args, slurm_handler):
    lm_dataloader, _, _ = model_config["data"]
    criterion = benchmark_config["criterion"]
    vocab_size = model_specs["vocab_size"]
    optimizer = model_config["optimizer"]

    if not args.benchmark_eval:
        model.train()
    log_number_of_parameters(model)

    total_loss = 0.0

    optimizer = optimizer(model.parameters())
    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] optimizer')

    total_tokens = 0
    total_tokens_per_log_interval = 0
    start_time = time.time()
    epoch_start_time = 0.0

    def get_batch(source):
        seq_len = len(source) - 1
        data = source[0:seq_len]
        target = source[1 : 1 + seq_len]
        return data, target
    
    for i, batch in enumerate(lm_dataloader):

        if i == 1:
            epoch_start_time = time.time()

        batch_start_time = time.time()

        source, target = get_batch(batch)
        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Get batch')

        if i > 0:
            total_tokens += source.numel()

        if args.benchmark_eval:
            input = source.cuda()
            target = target.cuda()
            output = model(input)
            loss = torch.nn.CrossEntropyLoss()(output.view(-1, vocab_size), target.view(-1))
        else:
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

        log_interval = 1
        total_tokens_per_log_interval += source.numel()
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            if dist.get_rank() == 0:
                _logger.info(
                    "| batch {:5d} | wps {:5.2f} | loss {:5.2f} | ppl {:8.2f}".format(
                        i, total_tokens_per_log_interval / elapsed, cur_loss, math.exp(cur_loss)
                    )
                )
            total_tokens_per_log_interval = 0
            total_loss = 0
            start_time = time.time()
        

        ### Notify DeepShare profiler about iteration training status
        batch_time = time.time() - batch_start_time
        if slurm_handler.profiler != None:
            slurm_handler.profiler.step(samples=args.batch_size * args.world_size, bt=batch_time)

    #end for

    if epoch_start_time != 0:
        torch.cuda.synchronize()
        wps = total_tokens / (time.time() - epoch_start_time)
    else:
        raise RuntimeError(
            "Unable to benchmark on a single batch. Increase the size " " of the dataset and rerun the benchmark."
        )
    return wps, loss.item()


def get_number_of_words(data):
    return data.size()[0] * data.size()[1]


def benchmark_language_model(model_config, model, benchmark_config, model_specs, args, slurm_handler):

    # Wait for another job
    if args.wait_workers != 0:
        slurm_handler.file_log(args.polling_file_path, 'r', mode='a')
        slurm_handler.file_poll(args.polling_file_path, 'r' * args.wait_workers)
        slurm_handler.print_with_rank('polling done!')
    
    epochs = benchmark_config["epochs"]
    if dist.get_rank() == 0:
        _logger.info(f'Total epochs: {epochs}')

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


def create_benchmark_config(model_name):
    """Return a dict with configurations required for benchmarking `model_name` model."""

    if model_name == "lm":
        return lm_wikitext2.get_benchmark_config()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def get_model_specs(model_name):
    """Return a dict with configurations required for configuring `model_name` model."""

    if model_name == "lm":
        return lm_wikitext2.get_model_config()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def get_golden_config(model_name, args):
    """Return a dict with the golden data for throughput and memory usage."""

    if model_name == "lm":
        return lm_wikitext2.get_golden_synthetic_stats()
    else:
        raise RuntimeError("Unrecognized args.model_mame " % args.model_name)


def benchmark_fsdp(args, slurm_handler):

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
    benchmark_config = create_benchmark_config(args.model_name)
    benchmark_config['epochs'] = args.epochs
    benchmark_config['batch_size'] = args.batch_size

    model_specs = get_model_specs(args.model_name)
    model_config = create_model_config(args, benchmark_config=benchmark_config, model_specs=model_specs)
    model = model_config["model"]
    config = {}

    if args.full_fp16:
        config["compute_dtype"] = torch.float16
        config["mixed_precision"] = False
    if args.flatten_parameters:
        config['flatten_parameters'] = True 
    if args.full_fp16:
        model = model.half()

    if args.enable_auto_wrap:
        with enable_wrap(wrapper_cls=FSDP, **config):
            fsdp_model = auto_wrap(model, auto_wrap_policy=default_auto_wrap_policy)
            fsdp_model = FSDP(fsdp_model, **config)
    else:
        fsdp_model = FSDP(model, **config)

    benchmark_language_model(model_config, fsdp_model, benchmark_config, model_specs, args, slurm_handler)


parser = argparse.ArgumentParser(description="benchmark")
parser.add_argument("--use_synthetic_data", action="store_true", help="Uses synthetic data for running benchmarks.")
parser.add_argument(
    "--model_name",
    default="lm",
    help="Language Model(LM) used to benchmark FSDP.",
)
parser.add_argument("--enable_auto_wrap", action="store_true", default=False, help="Use auto_wrap with FSDP")
parser.add_argument("--benchmark_eval", action="store_true", default=False, help="Benchmark evaluation workflow.")
parser.add_argument("--full_fp16", action="store_true", default=True, help="Benchmark in full fp16 mode (default: True)")

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
parser.add_argument('--flatten_parameters', action='store_true', default=False,
                    help='profile with torch profiler')                    
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

    _logger.info(f"Running FSDP benchmark with args: {args}")

    slurm_handler = DeepShareSlurmHandler(DummyDeepShareJobCheckpointer()) # TODO: add checkpointer
    if args.profile_gpu or args.profile_cpu or args.profile_progress:
        slurm_handler.install_profiler(profile_path=args.profile_path, profile_gpu=args.profile_gpu, 
                                        profile_cpu=args.profile_cpu, profile_progress=args.profile_progress, 
                                        profile_iteration=args.profile_iteration)
    
    # Train entry
    benchmark_fsdp(args, slurm_handler)