#!/bin/bash

# timm graphsage fsdp transformer dlrm moe

LOG_FILE=/home/gajagajago/deepshare/slurm_examples/out/0423_log
touch $LOG_FILE

#CS test
#Wait
for x in timm
do
	for x_place in 2,2 4,2 4,4 8,2 8,4 
	do
		for y in timm graphsage fsdp transformer dlrm moe
		do
			for y_place in 2,2 4,2 4,4 8,2 8,4 
			do
				echo "${x}_${x_place}_${y}_${x_place}" >> $LOG_FILE
				sbatch ${x}_${x_place}.sh >> $LOG_FILE
				sbatch ${y}_${y_place}.sh >> $LOG_FILE
				sleep 5m
				scancel --user=gajagajago
			done
		done
	done
done

for x in graphsage
do
	for x_place in 2,2 4,2 4,4 8,2 8,4 
	do
		for y in graphsage fsdp transformer dlrm moe
		do
			for y_place in 2,2 4,2 4,4 8,2 8,4 
			do
				echo "${x}_${x_place}_${y}_${x_place}" >> $LOG_FILE
				sbatch ${x}_${x_place}.sh >> $LOG_FILE
				sbatch ${y}_${y_place}.sh >> $LOG_FILE
				sleep 5m
				scancel --user=gajagajago
			done
		done
	done
done

for x in fsdp
do
	for x_place in 2,2 4,2 4,4 8,2 8,4 
	do
		for y in fsdp transformer dlrm moe
		do
			for y_place in 2,2 4,2 4,4 8,2 8,4 
			do
				echo "${x}_${x_place}_${y}_${x_place}" >> $LOG_FILE
				sbatch ${x}_${x_place}.sh >> $LOG_FILE
				sbatch ${y}_${y_place}.sh >> $LOG_FILE
				sleep 5m
				scancel --user=gajagajago
			done
		done
	done
done

for x in transformer
do
	for x_place in 2,2 4,2 4,4 8,2 8,4 
	do
		for y in transformer dlrm moe
		do
			for y_place in 2,2 4,2 4,4 8,2 8,4 
			do
				echo "${x}_${x_place}_${y}_${x_place}" >> $LOG_FILE
				sbatch ${x}_${x_place}.sh >> $LOG_FILE
				sbatch ${y}_${y_place}.sh >> $LOG_FILE
				sleep 5m
				scancel --user=gajagajago
			done
		done
	done
done

for x in dlrm
do
	for x_place in 2,2 4,2 4,4 8,2 8,4 
	do
		for y in dlrm moe
		do
			for y_place in 2,2 4,2 4,4 8,2 8,4 
			do
				echo "${x}_${x_place}_${y}_${x_place}" >> $LOG_FILE
				sbatch ${x}_${x_place}.sh >> $LOG_FILE
				sbatch ${y}_${y_place}.sh >> $LOG_FILE
				sleep 5m
				scancel --user=gajagajago
			done
		done
	done
done

for x in moe
do
	for x_place in 2,2 4,2 4,4 8,2 8,4 
	do
		for y in moe
		do
			for y_place in 2,2 4,2 4,4 8,2 8,4 
			do
				echo "${x}_${x_place}_${y}_${x_place}" >> $LOG_FILE
				sbatch ${x}_${x_place}.sh >> $LOG_FILE
				sbatch ${y}_${y_place}.sh >> $LOG_FILE
				sleep 5m
				scancel --user=gajagajago
			done
		done
	done
done