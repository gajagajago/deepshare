import argparse
from datetime import datetime
import logging
import glob
import os
from rl_training.policy_factory import PolicyFactory
from cluster_env import ClusterEnv

from datetime import datetime
date_time_ymd = datetime.now().strftime("%Y-%m-%d")

# DeepShare: Initialized global `_logger` and `args`.
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
args = None

def _parse_args():
    parser = argparse.ArgumentParser(description="GPU scheduler scheduler")

    parser.add_argument("--nodes", type=int, default=4,
                        help="number of nodes")
    parser.add_argument("--gpus-per-node", type=int, default=8,
                        help="number of GPUs per node")
    parser.add_argument("--total-jobsets", type=int, default=1,
                        help="Total number of jobsets to train")
    parser.add_argument("--episodes-per-jobset", type=int, default=1,
                        help="Total number of jobsets to train")
    parser.add_argument("--round-dur", type=int, default=300,
                        help="Length of one scheduling round in seconds")
    parser.add_argument("--rl-algo", type=str, default="PPO",
                        help="RL algorithms to use in training")
    parser.add_argument("--coeff-cont", type=float, default=0.8,
                        help="Reward weighting factor for average contention sensitivity (w1)")
    parser.add_argument("--coeff-util", type=float, default=0.2,
                        help="Reward weighting factor for cluster-wide utilization (w2)")
    parser.add_argument("--trace", default=f"{os.environ['DEEPSHARE_PATH']}/scheduler/traces/train", type=str, metavar="PATH",
                        help="path to job traces")
    parser.add_argument("--ckpt-dir", default=f"{os.environ['DEEPSHARE_PATH']}/scheduler/checkpoints/{date_time_ymd}", type=str, metavar="PATH",
                        help="path to policy checkpoints")
    parser.add_argument("--isolated-thp-path", default=f"{os.environ['DEEPSHARE_PATH']}/scheduler/jobs/isolated_thp.csv", type=str, metavar="PATH",
                        help="path to job throughputs in isolated cluster")
    parser.add_argument("--shared-thp-path", default=f"{os.environ['DEEPSHARE_PATH']}/scheduler/jobs/shared_thp_ratio.csv", type=str, metavar="PATH",
                        help="path to job throughputs in shared cluster")

    return parser.parse_args()


def cleanup():
    extensions = ["*.log"]
    for e in extensions:
        for f in glob.glob(e):
            os.remove(f)
            _logger.debug(f"Removed {f}")


if __name__ == "__main__":
    cleanup()

    args = _parse_args()

    ckpt_path_exists = os.path.exists(args.ckpt_dir)
    if not ckpt_path_exists:
        os.makedirs(args.ckpt_dir)
        _logger.info(f"Checkpoint dir: {args.ckpt_dir} created")

    # Init environment
    env = ClusterEnv(nr_node=args.nodes, gpus_per_node= args.gpus_per_node, round_dur=args.round_dur,
                    rl_algo=args.rl_algo, coeff_cont=args.coeff_cont, coeff_util=args.coeff_util, 
                    total_jobsets=args.total_jobsets, episodes_per_jobset=args.episodes_per_jobset,
                    trace=args.trace, ckpt_dir=args.ckpt_dir,
                    isolated_thp_path=args.isolated_thp_path, shared_thp_path=args.shared_thp_path)

    # Create RL policy with rl_algo
    policy = PolicyFactory().create_policy(env, args.rl_algo)

    # Setup and start training
    env.start(train=True, model=policy)
