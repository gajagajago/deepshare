#!/bin/bash

#SBATCH -J graphsageddp

#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1

#SBATCH --signal=USR2@11

#SBATCH --mem-per-gpu=10000
#SBATCH --cpus-per-task=3

#SBATCH --oversubscribe

#SBATCH --output=/home/gajagajago/deepshare/slurm_examples/out/%j.out

# options
declare -i WORLD_SIZE=4 # Number of processes participating in the job(https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)
CONDA_ENV=gnn
REQUIREMENTS="gnn_requirements.txt"

srun $DEEPSHARE_PATH/slurm_examples/setup.sh -c $CONDA_ENV -r $DEEPSHARE_PATH/slurm_examples/$REQUIREMENTS

# Path to write and read assigned master port
# Assumption: /job_addr path is already made on HDFS
srun mkdir -p $HADOOP_DIR/job_addr
export JOB_MASTER_PORT_LOCAL_PATH=$HADOOP_DIR/job_addr
export JOB_MASTER_PORT_HDFS_PATH=/job_addr

# Path to save local checkpoint (point of resume after preemption)
srun mkdir -p $HADOOP_DIR/local_checkpoint
LOCAL_CHECKPOINT_PATH=$HADOOP_DIR/local_checkpoint/$SLURM_JOBID

# Path to save HDFS checkpoint
# Assumption: /hdfs_checkpoint path is already made on HDFS
HDFS_CHECKPOINT_DIR=/hdfs_checkpoint

# Activate conda env
. $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Profiler
# Assumption: profile iteration >= 3 (for wait & warmup)
TENSORBOARD_PATH=$HADOOP_DIR/log/$SLURM_JOBID
srun mkdir -p $TENSORBOARD_PATH
declare -i PROFILE_ITERATION=64*$WORLD_SIZE

export WORLD_SIZE=$WORLD_SIZE

# Inter-job worker waiting
declare -i WAIT_WORKERS=2 # 1 more than my workers per node
POLLING_FILE_PATH=$DEEPSHARE_PATH/slurm_examples/out/ready
srun truncate -s 0 $POLLING_FILE_PATH

export NCCL_IB_DISABLE=1
srun -u python \
	$DEEPSHARE_PATH/slurm_examples/graphsage_ddp.py \
	--data-path=/cmsdata/ssd0/cmslab/dlcm_data/graph-data/Reddit \
	--batch-size=1024 \
	--resume=$LOCAL_CHECKPOINT_PATH \
	--hdfs-ckpt-dir=$HDFS_CHECKPOINT_DIR \
	--profile-path=$TENSORBOARD_PATH \
	--profile-progress \
	--profile-iteration=$PROFILE_ITERATION \
	--epochs 300 \
	--wait-workers=$WAIT_WORKERS \
	--polling-file-path=$POLLING_FILE_PATH \
	# --debug