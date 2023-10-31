# coding: utf-8

""" Modifications for DeepShare
This script was modified from original transformer-xl DataParallel code to PyTorch DistributedDataParallel.
(https://github.com/kimiyoung/transformer-xl)
"""

# DeepShare: Additional required packages
import logging
import os
import sys

# DeepShare: Add transformer-xl modules to system path
sys.path.append(os.path.join(os.environ['DEEPSHARE_PATH'], 'models/transformer-xl/pytorch'))
sys.path.append(os.path.join(os.environ['DEEPSHARE_PATH'], 'models/transformer-xl/pytorch/util'))

from utils.deepshare_handler import DeepShareSlurmHandler
from utils.checkpoint import DummyDeepShareJobCheckpointer

import argparse
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_utils import get_lm_corpus, LMOrderedIterator
from mem_transformer import MemTransformerLM


# DeepShare: Initialized global `_logger` and `args`.
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger('train')
args = None


def _parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')

    parser.add_argument('--data', type=str, default='../data/wikitext-103',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='wt103',
                        choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    parser.add_argument('--n_layer', type=int, default=12,
                        help='number of total layers')
    parser.add_argument('--n_head', type=int, default=10,
                        help='number of heads')
    parser.add_argument('--d_head', type=int, default=50,
                        help='head dimension')
    parser.add_argument('--d_embed', type=int, default=-1,
                        help='embedding dimension')
    parser.add_argument('--d_model', type=int, default=500,
                        help='model dimension')
    parser.add_argument('--d_inner', type=int, default=1000,
                        help='inner dimension in FF')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='global dropout rate')
    parser.add_argument('--dropatt', type=float, default=0.0,
                        help='attention probability dropout rate')
    parser.add_argument('--init', default='normal', type=str,
                        help='parameter initializer to use.')
    parser.add_argument('--emb_init', default='normal', type=str,
                        help='parameter initializer to use.')
    parser.add_argument('--init_range', type=float, default=0.1,
                        help='parameters initialized by U(-init_range, init_range)')
    parser.add_argument('--emb_init_range', type=float, default=0.01,
                        help='parameters initialized by U(-init_range, init_range)')
    parser.add_argument('--init_std', type=float, default=0.02,
                        help='parameters initialized by N(0, init_std)')
    parser.add_argument('--proj_init_std', type=float, default=0.01,
                        help='parameters initialized by N(0, init_std)')
    parser.add_argument('--optim', default='adam', type=str,
                        choices=['adam', 'sgd', 'adagrad'],
                        help='optimizer to use.')
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='initial learning rate (0.00025|5 for adam|sgd)')
    parser.add_argument('--mom', type=float, default=0.0,
                        help='momentum for sgd')
    parser.add_argument('--scheduler', default='cosine', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                        help='lr scheduler to use.')
    parser.add_argument('--warmup_step', type=int, default=0,
                        help='upper epoch limit')
    parser.add_argument('--decay_rate', type=float, default=0.5,
                        help='decay factor when ReduceLROnPlateau is used')
    parser.add_argument('--lr_min', type=float, default=0.0,
                        help='minimum learning rate during annealing')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--clip_nonemb', action='store_true',
                        help='only clip the gradient of non-embedding params')
    parser.add_argument('--max_step', type=int, default=100000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=60,
                        help='batch size')
    parser.add_argument('--batch_chunk', type=int, default=1,
                        help='split batch into chunks to save memory')
    parser.add_argument('--tgt_len', type=int, default=70,
                        help='number of tokens to predict')
    parser.add_argument('--eval_tgt_len', type=int, default=50,
                        help='number of tokens to predict for evaluation')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    parser.add_argument('--mem_len', type=int, default=0,
                        help='length of the retained previous heads')
    parser.add_argument('--not_tied', action='store_true',
                        help='do not tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--adaptive', action='store_true',
                        help='use adaptive softmax')
    parser.add_argument('--div_val', type=int, default=1,
                        help='divident value for adapative input and softmax')
    parser.add_argument('--pre_lnorm', action='store_true',
                        help='apply LayerNorm to the input instead of the output')
    parser.add_argument('--varlen', action='store_true',
                        help='use variable length')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='use multiple GPU')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--eval-interval', type=int, default=4000,
                        help='evaluation interval')
    parser.add_argument('--work_dir', default='LM-TFM', type=str,
                        help='experiment directory.')
    parser.add_argument('--restart', action='store_true',
                        help='restart training from the saved checkpoint')
    parser.add_argument('--restart_dir', type=str, default='',
                        help='restart dir')
    parser.add_argument('--same_length', action='store_true',
                        help='use the same attn length for all tokens')
    parser.add_argument('--attn_type', type=int, default=0,
                        help='attention type. 0 for ours, 1 for Shaw et al,'
                        '2 for Vaswani et al, 3 for Al Rfou et al.')
    parser.add_argument('--clamp_len', type=int, default=-1,
                        help='use the same pos embeddings after clamp_len')
    parser.add_argument('--eta_min', type=float, default=0.0,
                        help='min learning rate for cosine scheduler')
    parser.add_argument('--gpu0_bsz', type=int, default=-1,
                        help='batch size on gpu 0')
    parser.add_argument('--max_eval_steps', type=int, default=-1,
                        help='max eval steps')
    parser.add_argument('--sample_softmax', type=int, default=-1,
                        help='number of samples in sampled softmax')
    parser.add_argument('--patience', type=int, default=0,
                        help='patience')
    parser.add_argument('--finetune_v2', action='store_true',
                        help='finetune v2')
    parser.add_argument('--finetune_v3', action='store_true',
                        help='finetune v3')
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can '
                        'improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument'
                        ' supersedes --static-loss-scale.')

    # Custom argument added for DeepShare
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='set log level to loggig.debug')
    parser.add_argument('--hdfs-ckpt-dir', default='', type=str, metavar='PATH',
                        help='HDFS directory to store checkpoint (default: none)')
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
    parser.add_argument("--wait-workers", type=int, default=0,
                        help='how many workers to wait for getting prepared(including itself)')
    parser.add_argument('--polling-file-path', default='./out/ready', type=str, metavar='PATH',
                        help='path to polling file')

    args = parser.parse_args()
    args.tied = not args.not_tied

    if args.d_embed < 0:
        args.d_embed = args.d_model

    assert args.ext_len >= 0, 'extended context length must be non-negative'
    assert args.batch_size % args.batch_chunk == 0

    args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
    args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))

    return args


""" Setup seed, device """
def setup():    
    # Set the logging level
    if args.debug:
        _logger.info(f'Set log level to {logging.DEBUG}')
        _logger.setLevel(logging.DEBUG)

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            _logger.warning('You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed_all(args.seed)

    _logger.debug(f'Available devices: {torch.cuda.device_count()}')
    args.device = torch.device('cuda' if args.cuda else 'cpu')


def run(slurm_handler: DeepShareSlurmHandler):
    """ Setup DDP """
    if args.multi_gpu:
        args.local_rank = 0

        os.environ['MASTER_ADDR'] = f'{slurm_handler.master_addr}'
        os.environ['MASTER_PORT'] = f'{slurm_handler.master_port}'

        torch.distributed.init_process_group(backend="nccl", rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['WORLD_SIZE']))

        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)

        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

        _logger.debug(
            f"master address = {slurm_handler.master_addr}, "
            + f"master port = {slurm_handler.master_port}, "
            + f"[{os.getpid()}]: world_size = {torch.distributed.get_world_size()}, "
            + f"rank = {torch.distributed.get_rank()}, "
            + f"visibile devices={os.environ['CUDA_VISIBLE_DEVICES']} \n"
        )
    else:
        _logger.debug('Training with a single process on 1 GPUs.')


    """ Load data """
    corpus = get_lm_corpus(args.data, args.dataset)
    ntokens = len(corpus.vocab)
    args.n_token = ntokens
        
    # Distributes train data over `WORLD_SIZE` devices and creates an iterator per device
    corpus_len = len(corpus.train)
    tr_iter_start = corpus_len // int(os.environ['WORLD_SIZE']) * int(os.environ['SLURM_PROCID'])
    tr_iter_end = min(len(corpus.train) // int(os.environ['WORLD_SIZE']) * (int(os.environ['SLURM_PROCID']) + 1), corpus_len)
    tr_iter = LMOrderedIterator(corpus.train[tr_iter_start:tr_iter_end], args.batch_size, args.tgt_len,
        device=args.device, ext_len=args.ext_len)

    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] len(corpus.train): {corpus_len}')
    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] [{tr_iter_start}:{tr_iter_end})')
    
    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [False]
    if args.adaptive:
        assert args.dataset in ['wt103', 'lm1b']
        if args.dataset == 'wt103':
            cutoffs = [20000, 40000, 200000]
            tie_projs += [True] * len(cutoffs)
        elif args.dataset == 'lm1b':
            cutoffs = [60000, 100000, 640000]
            tie_projs += [False] * len(cutoffs)

    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Dataloader initialized')


    """ Build model """
    def init_weight(weight):
        if args.init == 'uniform':
            nn.init.uniform_(weight, -args.init_range, args.init_range)
        elif args.init == 'normal':
            nn.init.normal_(weight, 0.0, args.init_std)

    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                init_weight(m.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                init_weight(m.cluster_weight)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, args.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)

    def update_dropout(m):
        classname = m.__class__.__name__
        if classname.find('Dropout') != -1:
            if hasattr(m, 'p'):
                m.p = args.dropout

    def update_dropatt(m):
        if hasattr(m, 'dropatt'):
            m.dropatt.p = args.dropatt

    # TODO: DeepShare checkpointer restart entry for model states
    if args.restart:
        # Model load
        with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
            model = torch.load(f)
        model = model.float()
        model.apply(update_dropout)
        model.apply(update_dropatt)
    else:
        # Model creation
        model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
            args.d_head, args.d_inner, args.dropout, args.dropatt,
            tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
            tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
            ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
            same_length=args.same_length, attn_type=args.attn_type,
            clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)
        model.apply(weights_init)
        model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing
    args.n_all_param = sum([p.nelement() for p in model.parameters()])
    args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

    # Move model to local GPU 
    model = model.to(args.device)
    model = DDP(model, device_ids=[args.local_rank])

    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Model created and moved to GPU')

    """ Declare optimizer """
    optimizer_sparse = None
    if args.optim.lower() == 'sgd':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
            optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr,
                momentum=args.mom)
    elif args.optim.lower() == 'adam':
        if args.sample_softmax > 0:
            dense_params, sparse_params = [], []
            for param in model.parameters():
                if param.size() == model.word_emb.weight.size():
                    sparse_params.append(param)
                else:
                    dense_params.append(param)
            optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
            optimizer = optim.Adam(dense_params, lr=args.lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    # TODO: DeepShare checkpointer restart entry for optimizer states
    if args.restart:
        if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
            with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
                opt_state_dict = torch.load(f)
                optimizer.load_state_dict(opt_state_dict)
        else:
            print('Optimizer was not saved. Start from scratch.')

    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] Optimizer created')


    """ Declare scheduler """ 
    scheduler_sparse = None
    if args.scheduler == 'cosine':
        # here we do not set eta_min to lr_min to be backward compatible
        # because in previous versions eta_min is default to 0
        # rather than the default value of lr_min 1e-6
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
            args.max_step, eta_min=args.eta_min) # should use eta_min arg
        if args.sample_softmax > 0:
            scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
                args.max_step, eta_min=args.eta_min) # should use eta_min arg
    elif args.scheduler == 'inv_sqrt':
        # originally used for Transformer (in Attention is all you need)
        def lr_lambda(step):
            # return a multiplier instead of a learning rate
            if step == 0 and args.warmup_step == 0:
                return 1.
            else:
                return 1. / (step ** 0.5) if step > args.warmup_step \
                    else step / (args.warmup_step ** 1.5)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
        if args.sample_softmax > 0:
            scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
                factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
    elif args.scheduler == 'constant':
        pass

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

    """ Train epochs """
    try:
        for epoch in range(args.epochs):
            train_one_epoch(epoch, tr_iter, model, optimizer, scheduler, args, slurm_handler, optimizer_sparse=optimizer_sparse, scheduler_sparse=scheduler_sparse)
    except KeyboardInterrupt:
        _logger.debug('Exiting from training early')


def train_one_epoch(epoch, data_loader, model, optimizer, scheduler, args, slurm_handler, optimizer_sparse=None, scheduler_sparse=None):

    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] [Epoch {epoch}] Entered train_one_epoch()')

    # Turn on training mode which enables dropout.
    train_step = train_loss = 0

    model.train()
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()

    _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] [Epoch {epoch}] Steps: {data_loader.n_step}')
  
    last_idx = data_loader.n_step - 1

    for batch_idx, (data, target, _) in enumerate(data_loader):

        batch_start_time = time.time()

        ret = model(data, target, *mems)
        loss, mems = ret[0], ret[1:]
        loss = loss.float().mean().type_as(loss)
        loss.backward()
        train_loss += loss.float().item()

        _logger.debug(f'[Rank {os.environ["SLURM_PROCID"]}] [Batch {batch_idx}] trained')

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)
        
        ### Notify DeepShare profiler about iteration training status
        batch_time = time.time() - batch_start_time
        if slurm_handler.profiler != None:
            slurm_handler.profiler.step(samples=args.batch_size * args.world_size, bt=batch_time)
    
    # end for


if __name__ == '__main__':
    args = _parse_args()
    setup()
    slurm_handler = DeepShareSlurmHandler(DummyDeepShareJobCheckpointer()) # TODO: add checkpointer
    if args.profile_gpu or args.profile_cpu or args.profile_progress:
        slurm_handler.install_profiler(profile_path=args.profile_path, profile_gpu=args.profile_gpu, 
                                        profile_cpu=args.profile_cpu, profile_progress=args.profile_progress, 
                                        profile_iteration=args.profile_iteration)
    run(slurm_handler)