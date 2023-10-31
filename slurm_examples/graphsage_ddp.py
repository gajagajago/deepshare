import argparse
import copy
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

from utils.deepshare_handler import DeepShareSlurmHandler
from utils.checkpoint import DummyDeepShareJobCheckpointer


# DeepShare: Initialized global `_logger` and `args`.
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger('train')
args = None


class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all: Tensor, device: torch.device,
                  subgraph_loader: NeighborLoader) -> Tensor:

        pbar = tqdm(total=len(subgraph_loader) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.node_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                x = x[:batch.batch_size]
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
                pbar.update(1)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all


def run(slurm_handler):
    assert os.environ.get('WORLD_SIZE')
    os.environ['MASTER_ADDR'] = f'{slurm_handler.master_addr}'
    os.environ['MASTER_PORT'] = f'{slurm_handler.master_port}'
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['RANK'])
    args.local_rank = 0

    torch.distributed.init_process_group('nccl', rank=args.rank, world_size=args.world_size)

    _logger.debug(
        f"master address = {slurm_handler.master_addr}, "
        + f"master port = {slurm_handler.master_port}, "
        + f"[{os.getpid()}]: world_size = {torch.distributed.get_world_size()}, "
        + f"rank = {torch.distributed.get_rank()}, "
        + f"visibile devices={os.environ['CUDA_VISIBLE_DEVICES']} \n"
    )

    slurm_handler.print_with_rank("Reached 1") # DeepShare TMP

    dataset = Reddit(args.data_path)
    data = dataset[0].to(args.local_rank, 'x', 'y')  # Move to device for faster feature fetch.

    slurm_handler.print_with_rank("Reached 2") # DeepShare TMP

    # Split training indices into `world_size` many chunks:
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // args.world_size)[args.local_rank]
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    kwargs = dict(batch_size=args.batch_size, num_workers=num_workers, persistent_workers=True)
    train_loader = NeighborLoader(data, input_nodes=train_idx,
                                  num_neighbors=[25, 10], shuffle=True,
                                  drop_last=True, **kwargs)

    slurm_handler.print_with_rank("Reached 3") # DeepShare TMP

    if args.rank == 0:  # Create single-hop evaluation neighbor loader:
        subgraph_loader = NeighborLoader(copy.copy(data), num_neighbors=[-1],
                                         shuffle=False, **kwargs)
        # No need to maintain these features during evaluation:
        del subgraph_loader.data.x, subgraph_loader.data.y
        # Add global node index information:
        subgraph_loader.data.node_id = torch.arange(data.num_nodes)

    slurm_handler.print_with_rank("Reached 4") # DeepShare TMP

    torch.manual_seed(12345)
    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(args.local_rank)
    slurm_handler.print_with_rank(f"dataset.num_features[{dataset.num_features}] dataset.num_classes {dataset.num_classes}") # DeepShare TMP

    slurm_handler.print_with_rank("Reached 4-1") # DeepShare TMP
    slurm_handler.print_with_rank(f"local rank: {args.local_rank}")

    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    slurm_handler.print_with_rank("Reached 4-2") # DeepShare TMP

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

    slurm_handler.print_with_rank("Reached 5") # DeepShare TMP

    for epoch in range(args.epochs):

        model.train()
        for batch_idx, batch in enumerate(train_loader):

            batch_start_time = time.time()

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index.to(args.local_rank))[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()

            ### Notify DeepShare profiler about iteration training status
            batch_time = time.time() - batch_start_time
            if slurm_handler.profiler != None:
                slurm_handler.profiler.step(samples=args.batch_size * args.world_size, bt=batch_time)

        dist.barrier()

        if args.rank == 0:
            _logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        dist.barrier()

    dist.destroy_process_group()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed graph sampling with GraphSAGE"
    )
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
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
    parser.add_argument('--data-path', default='/cmsdata/ssd0/cmslab/dlcm_data/graph-data/Reddit', type=str, metavar='PATH',
                    help='path to graph data')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
    parser.add_argument('-b', '--batch-size', type=int, default=1024, metavar='N',
                        help='Input batch size for training (default: 8)')
    parser.add_argument("--wait-workers", type=int, default=0,
                        help='how many workers to wait for getting prepared(including itself)')
    parser.add_argument('--polling-file-path', default='./out/ready', type=str, metavar='PATH',
                        help='path to polling file')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = _parse_args()

    # DeepShare: Setup log level
    if args.debug:
        _logger.info(f'Set log level to {logging.DEBUG}')
        _logger.setLevel(logging.DEBUG)
    
    slurm_handler = DeepShareSlurmHandler(DummyDeepShareJobCheckpointer()) # TODO: add GNN checkpointer
    if args.profile_gpu or args.profile_cpu or args.profile_progress:
        slurm_handler.install_profiler(profile_path=args.profile_path, profile_gpu=args.profile_gpu, 
                                        profile_cpu=args.profile_cpu, profile_progress=args.profile_progress, 
                                        profile_iteration=args.profile_iteration)
    run(slurm_handler)