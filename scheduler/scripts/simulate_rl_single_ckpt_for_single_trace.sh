#!/bin/bash

SCHED=rl
TRACE=$1
CKPT=$2

TRUNC_CKPT=$(echo ${CKPT} | rev | cut -d "/" -f1 | rev)
echo $TRUNC_CKPT

python launch.py --ckpt $CKPT --simulate --trace $TRACE --scheduler $SCHED