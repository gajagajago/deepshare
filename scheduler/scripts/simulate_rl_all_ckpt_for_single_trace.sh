#!/bin/bash

# Simulate all checkpoints in a ckpt directory for performance comparison

SCHED=rl
TRACE=$1
CKPT_DIR=$2

for CKPT in $CKPT_DIR/* 
do
  # Truncate ckpt name from ckpt path
  TRUNC_CKPT=$(echo ${CKPT} | rev | cut -d "/" -f1 | rev)
  echo $TRUNC_CKPT

  python launch.py --ckpt $CKPT --simulate --trace $TRACE --scheduler $SCHED
done
