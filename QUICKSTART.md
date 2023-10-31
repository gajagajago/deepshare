## Quickstart

### Training with simulator

```
python3 ./train.py [--nodes NODES] [--gpus-per-node GPUS_PER_NODE] \
        [--total-jobsets TOTAL_JOBSETS_IN_TRACE] [--episodes-per-jobset EPISODES_PER_JOBSET] \
        [--round-dur ROUND_DURATION] [--rl-algo RL_ALGORITHM] \
        [--coeff-cont W1] [--coeff-util W2] \
        [--trace TRACE_DIR] [--ckpt-dir CKPT_DIR] \
        [--isolated-thp-path ISOLATED_THP_PATH] [--shared-thp-path SHARED_THP_PATH]
```

### Deploying on real GPU Cluster

```
python3 ./launch.py [--nodes NODES] [--gpus-per-node GPUS_PER_NODE] \
        [--round-dur ROUND_DURATION] [--scheduler SCHEDULER] [--simulate] \
        [--trace TRACE_DIR] [--ckpt CKPT] \
        [--isolated-thp-path ISOLATED_THP_PATH] [--shared-thp-path SHARED_THP_PATH]
```