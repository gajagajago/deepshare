#!/bin/bash

#SBATCH -J dlrm

#SBATCH --nodes=4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node 2
#SBATCH --gpus-per-task 1

#SBATCH --signal=USR2@11

#SBATCH --mem-per-gpu=10000
#SBATCH --cpus-per-task=3

#SBATCH --oversubscribe

#SBATCH --output=/home/gajagajago/deepshare/slurm_examples/out/%j.out

# options
declare -i WORLD_SIZE=8 # Number of processes participating in the job(https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)
CONDA_ENV=dlrm
REQUIREMENTS="dlrm_requirements.txt"

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
declare -i PROFILE_ITERATION=128*$WORLD_SIZE

export WORLD_SIZE=$WORLD_SIZE

# Inter-job worker waiting
declare -i WAIT_WORKERS=3 # 1 more than my workers per node
POLLING_FILE_PATH=$DEEPSHARE_PATH/slurm_examples/out/ready
srun truncate -s 0 $POLLING_FILE_PATH

export NCCL_IB_DISABLE=1
srun -u python \
	$DEEPSHARE_PATH/slurm_examples/dlrm_ddp.py \
    --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" \
    --data-generation=dataset --data-set=kaggle \
    --raw-data-file=/cmsdata/ssd0/cmslab/dlcm_data/dlrm_data/criteo_kaggle/train.txt \
    --processed-data-file=/cmsdata/ssd0/cmslab/dlcm_data/dlrm_data/criteo_kaggle/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=48 \
    --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 \
    --nepochs=100 --use-gpu \
	--accumulate-iteration=1 \
    --profile-path=$TENSORBOARD_PATH \
	--profile-progress \
	--profile-iteration=$PROFILE_ITERATION \
	--wait-workers=$WAIT_WORKERS \
	--polling-file-path=$POLLING_FILE_PATH \
    # --debug