#!/bin/bash

#SBATCH -J timmddp

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1

#SBATCH --signal=USR2@11

#SBATCH --mem-per-gpu=10000
#SBATCH --cpus-per-task=3

#SBATCH --oversubscribe

#SBATCH --output=/home/gajagajago/deepshare/slurm_examples/out/%j.out

# options
declare -i WORLD_SIZE=2 # Number of processes participating in the job(https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)
CONDA_ENV=timm_ddp 
REQUIREMENTS="timm_requirements.txt"

srun $DEEPSHARE_PATH/slurm_examples/setup.sh -c $CONDA_ENV -r $DEEPSHARE_PATH/slurm_examples/$REQUIREMENTS

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
declare -i PROFILE_ITERATION=512*$WORLD_SIZE

export WORLD_SIZE=$WORLD_SIZE

# Inter-job worker waiting
declare -i WAIT_WORKERS=0
POLLING_FILE_PATH=$DEEPSHARE_PATH/slurm_examples/out/ready
srun truncate -s 0 $POLLING_FILE_PATH

export NCCL_IB_DISABLE=1
srun -u python \
	$DEEPSHARE_PATH/slurm_examples/timm_ddp.py \
	/cmsdata/ssd0/cmslab/imagenet-pytorch \
	--model=mobilenetv3_small_075 --batch-size=16 \
	--output /tmp \
	--accumulate-iteration=1 \
	--resume=$LOCAL_CHECKPOINT_PATH \
	--hdfs-ckpt-dir=$HDFS_CHECKPOINT_DIR \
  --profile-path=$TENSORBOARD_PATH \
	--profile-progress \
	--profile-iteration=$PROFILE_ITERATION \
	--wait-workers=$WAIT_WORKERS \
	--polling-file-path=$POLLING_FILE_PATH \
	# --debug