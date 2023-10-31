# Slurm Configurations

## Prerequisites

`$SLURM_CONF_DIR` should be set to the path to Slurm configuration files. Slurm will try to find its `slurm.conf`, `gres.conf`, `cgroup.conf`, etc., under this path. `SLURM_CONF` should be set to the path to `slurm.conf` under `$SLURM_CONF_DIR` according to [`controller.c`](/slurm/src/slurmctld/controller.c/#L303).

```
export $SLURM_CONF_DIR = {PATH_TO_SLURM_CONF_FILES} # $DEEPSHARE_PATH/slurm/etc
export $SLURM_CONF = {PATH_TO_SLURM_CONF} # $SLURM_CONF_DIR/slurm.conf
```

## `slurm.conf`

General Slurm configuration information about node management, partitions, and scheduling parameters. 

Our Slurm scheduler is configured as Round-robin preemption scheduler. 
- [Partition based multi-factor priority policy](https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1594263523/Job+Priority): Jobs' priority is calculated from multiple factors, heavily depending on the priority of its partition.
- Requeue preemption mechanism: Jobs are requeued when preempted, while releasing all its affinitized resources. 
- Round-robin scheduling: Resource (re)allocation is triggered every 30 seconds. Round interval can be set using `sched_min_interval`.

## `gres.conf`

Generic resource configuration informaion on each compute node.

### Test

Submit two `srun` scripts sequentially. Jobs will be submitted as `PENDING` state, and be converted to `RUNNING` state after being allocated resources at the round interval.

```
srun --oversubscribe sleep 100 &
srun --oversubscribe sleep 100 &
```

Submit two `srun` scripts that require all gres(6 GPUs for elsa-11) in the node sequentially. The job in the higher priority partition will preempt the job in the lower priority partition.

```
sbatch -p prio-low hold_gpu.sh
sbatch -p prio-high hold_gpu.sh
```
