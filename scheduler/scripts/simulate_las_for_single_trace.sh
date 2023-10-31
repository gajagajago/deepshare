#!/bin/bash

SCHED=las
TRACE=$1

python launch.py --simulate --trace $TRACE --scheduler $SCHED