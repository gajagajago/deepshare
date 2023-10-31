import argparse
import os
import glob
import logging

from stable_baselines3 import PPO
from simulator.cluster_env import ClusterEnv


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _parse_args():
  parser = argparse.ArgumentParser(description="GPU scheduler simulator")

  parser.add_argument("--nodes", type=int, default=4,
                      help="number of nodes")
  parser.add_argument("--gpus-per-node", type=int, default=8,
                      help="number of GPUs per node")
  parser.add_argument("--job-trace", default=f"{os.environ['DEEPSHARE_PATH']}/simulator/traces/simulate/0/trace", type=str, metavar="PATH",
                      help="path to job trace file")
  parser.add_argument("--model-checkpoint", default="/cmsdata/ssd0/cmslab/dlcm_data/rl-dlcm-ckpt", type=str, metavar="PATH",
                      help="path to model ckpt file")
  parser.add_argument("--scheduler", default="rl", type=str,
                      help="schduler to use <rl/srtf/las>")
  parser.add_argument('--simulate', action='store_true', default=False,
                    help='whether to simulate the inference procedure')
  return parser.parse_args()


def cleanup():
  for f in glob.glob("*.log"):
      os.remove(f)

# Inference entrypoint
# Inference can be done in either simulated/real environment
if __name__ == "__main__":
  args = _parse_args()
  cleanup()

  env = ClusterEnv(args.nodes, args.gpus_per_node, trace=args.job_trace)

  if args.scheduler == "rl":
    model = PPO.load(args.model_checkpoint, env=env)
  elif args.scheduler == "srtf" or args.scheduler == "las":
    model = None
  else:
    _logger.info("Invalid scheduler")

  env.start(inference=True, simulated_inference=args.simulate, model=model, scheduler=args.scheduler)