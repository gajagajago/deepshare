import argparse
import os
import glob
import logging

from stable_baselines3 import PPO
from scheduler.cluster_env import ClusterEnv

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _parse_args():
  parser = argparse.ArgumentParser(description="DeepShare scheduler")

  parser.add_argument("--nodes", type=int, default=4,
                      help="number of nodes")
  parser.add_argument("--gpus-per-node", type=int, default=8,
                      help="number of GPUs per node")
  parser.add_argument("--round-dur", type=int, default=300,
                      help="Length of one scheduling round in seconds")
  parser.add_argument("--trace", default=f"{os.environ['DEEPSHARE_PATH']}/scheduler/traces/simulate/trace0", type=str, metavar="PATH",
                      help="path to job traces")
  parser.add_argument("--scheduler", default="rl", type=str,
                      help="schduler to use <rl/srtf/las>")
  parser.add_argument("--ckpt", default=f"{os.environ['DEEPSHARE_PATH']}/scheduler/checkpoints/scheduler/checkpoints/contrl_w1_0.3_w2_0.7", type=str, metavar="PATH",
                      help="path to RL policy ckpt file")
  parser.add_argument('--simulate', action='store_true', default=False,
                    help='whether to simulate the scheduling procedure')
  parser.add_argument("--isolated-thp-path", default=f"{os.environ['DEEPSHARE_PATH']}/scheduler/jobs/isolated_thp.csv", type=str, metavar="PATH",
                      help="path to job throughputs in isolated cluster")
  parser.add_argument("--shared-thp-path", default=f"{os.environ['DEEPSHARE_PATH']}/scheduler/jobs/shared_thp_ratio.csv", type=str, metavar="PATH",
                      help="path to job throughputs in shared cluster")
  return parser.parse_args()


def cleanup():
  for f in glob.glob("*.log"):
      os.remove(f)

# Scheduler entrypoint
# Scheduling can be executed in either simulated/real environment
if __name__ == "__main__":
  args = _parse_args()
  cleanup()

  # Initialize cluster environment
  # For real GPU cluster scheduling, arguments must match the real cluster spec.
  env = ClusterEnv(nr_node=args.nodes, gpus_per_node=args.gpus_per_node, round_dur=args.round_dur,
                  trace=args.trace, isolated_thp_path=args.isolated_thp_path, shared_thp_path=args.shared_thp_path)

  # Scheduling policy plugin
  if args.scheduler == "rl":
    model = PPO.load(args.ckpt, env=env)
  elif args.scheduler == "srtf" or args.scheduler == "las":
    model = None
  else:
    _logger.info("Invalid scheduler")

  # Start scheduling
  avg_stp, avg_jct, avg_util = env.start(train=False, simulate=args.simulate, model=model, scheduler=args.scheduler)

  # Log scheduling performance
  _logger.info(f"Avg. JCT: {avg_jct}\n"
  f"Avg. Util: {avg_util}\n"
  f"Avg. System Thp: {avg_stp}\n")