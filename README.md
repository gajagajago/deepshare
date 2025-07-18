# Network Contention-Aware Cluster Scheduling with Reinforcement Learning [[Paper](https://arxiv.org/abs/2310.20209)] [[Poster](assets/deepshare_poster.pdf)]

DeepShare presents (1) a method to mitigate network contention by training a sophisticated scheduling policy using RL and (2) a framework to deploy it for efficient management of distributed DL jobs in GPU clusters.

<img src="assets/exp.png" title="exp">

TLDR: Distributed DL training on shared GPU clusters is prone to network contention between training jobs. This is because existing schedulers mainly focus on allocation of <strong>dedicated</strong> computation resources (e.g., GPU) but are often agnostic to <strong>shared</strong> network resources (e.g., PCIe, NVLink, and Infiniband). This can be addressed by incorporating a contention-aware scheduler that dynamically schedules and migrates jobs according to cluster-wide network contention. DeepShare presents an end-to-end system for training such efficient scheduling policies with RL to its deployment on GPU clusters. Scheduling policies trained with DeepShare (RL-base and RL-hybrid in above figure) show that training latency is improved by up to 20.7% compared to state-of-the-art schedulers.

<img src="assets/system.png" title="system">

## Getting started

- Refer to [Installation](INSTALL.md) for complete instructions on environment setup and installation.
- Refer to [Quickstart](QUICKSTART.md) for training scheduling policies with RL and deploying on GPU cluster.
- Refer to [Examples](slurm_examples/EXAMPLE.md) for writing custom job scripts.

## Citation
```
@inproceedings{ryu2023network,
  title={Network contention-aware cluster scheduling with reinforcement learning},
  author={Ryu, Junyeol and Eo, Jeongyoon},
  booktitle={2023 IEEE 29th International Conference on Parallel and Distributed Systems (ICPADS)},
  pages={2742--2745},
  year={2023},
  organization={IEEE}
}
```

## Contact

Junyeol Ryu (jyeol.ryu@gmail.com)
