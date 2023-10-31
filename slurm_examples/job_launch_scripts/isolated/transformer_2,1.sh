#!/bin/bash

#SBATCH -J transformer-xl

#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node 2
#SBATCH --gpus-per-task 1

#SBATCH --signal=USR2@11

#SBATCH --mem-per-gpu=10000
#SBATCH --cpus-per-task=3

#SBATCH --oversubscribe

#SBATCH --output=/home/gajagajago/deepshare/slurm_examples/out/%j.out

# options
declare -i WORLD_SIZE=2 # Number of processes participating in the job(https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)
CONDA_ENV=transformer-xl
REQUIREMENTS="transformer-xl_requirements.txt"

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
declare -i PROFILE_ITERATION=4*$WORLD_SIZE

export WORLD_SIZE=$WORLD_SIZE

# Inter-job worker waiting
declare -i WAIT_WORKERS=0
POLLING_FILE_PATH=$DEEPSHARE_PATH/slurm_examples/out/ready
srun truncate -s 0 $POLLING_FILE_PATH

# transformer-xl configs
DATA_PATH=/cmsdata/ssd0/cmslab/dlcm_data/Transformer-XL-data/wikitext-103
declare -i MEM_LEN=0 # Else, raises StopIteration exception

export NCCL_IB_DISABLE=1
srun -u python \
        $DEEPSHARE_PATH/slurm_examples/transformer-xl_ddp.py \
        --cuda \
        --data $DATA_PATH \
        --dataset wt103 \
        --adaptive \
        --n_layer 36 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 200000 \
        --tgt_len 150 \
        --mem_len $MEM_LEN \
        --eval_tgt_len 150 \
        --batch_size 16 \
        --multi_gpu \
        --gpu0_bsz 4 \
        --work_dir /tmp/transformer-xl \
        --epochs 100 \
        --profile-path=$TENSORBOARD_PATH \
        --profile-progress \
        --profile-iteration=$PROFILE_ITERATION \
        --wait-workers=$WAIT_WORKERS \
        --polling-file-path=$POLLING_FILE_PATH \
        # --debug