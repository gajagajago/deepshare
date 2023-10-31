#!/bin/bash

SCHED=srtf
TRACE=$1

python launch.py --simulate --trace $TRACE --scheduler $SCHED