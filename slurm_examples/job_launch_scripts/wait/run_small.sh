#!/bin/bash

# timm graphsage fsdp transformer dlrm moe

LOG_FILE=/home/gajagajago/deepshare/slurm_examples/out/0423_log_small
touch $LOG_FILE

#CS test
#Wait
#At each interval let two sets pass

declare -i even=0
for x in timm
do
 	for x_place in 2,2 4,2 8,2 
	do
		for y in timm graphsage fsdp transformer dlrm moe
		do
			for y_place in 2,2 4,2 8,2
			do
				echo "${x}_${x_place}_${y}_${y_place}" >> $LOG_FILE

        if [ $(expr $even % 2) == "0" ]
        then
          sbatch --nodelist=elsa-10,elsa-11 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-10,elsa-11 ${y}_${y_place}.sh >> $LOG_FILE
        else
          sbatch --nodelist=elsa-13,elsa-15 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-13,elsa-15 ${y}_${y_place}.sh >> $LOG_FILE 
				  sleep 5m
					scancel --user=gajagajago # cancel all job
					sleep 30s
        fi
				even+=1
			done
		done
	done
done

declare -i even=0
for x in graphsage
do
 	for x_place in 2,2 4,2 8,2 
	do
		for y in graphsage fsdp transformer dlrm moe
		do
			for y_place in 2,2 4,2 8,2
			do
				echo "${x}_${x_place}_${y}_${y_place}" >> $LOG_FILE

        if [ $(expr $even % 2) == "0" ]
        then
          sbatch --nodelist=elsa-10,elsa-11 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-10,elsa-11 ${y}_${y_place}.sh >> $LOG_FILE
        else
          sbatch --nodelist=elsa-13,elsa-15 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-13,elsa-15 ${y}_${y_place}.sh >> $LOG_FILE 
				  sleep 5m
					scancel --user=gajagajago # cancel all job
					sleep 30s
        fi
				even+=1
			done
		done
	done
done

declare -i even=0
for x in fsdp
do
 	for x_place in 2,2 4,2 8,2 
	do
		for y in fsdp transformer dlrm moe
		do
			for y_place in 2,2 4,2 8,2
			do
				echo "${x}_${x_place}_${y}_${y_place}" >> $LOG_FILE

        if [ $(expr $even % 2) == "0" ]
        then
          sbatch --nodelist=elsa-10,elsa-11 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-10,elsa-11 ${y}_${y_place}.sh >> $LOG_FILE
        else
          sbatch --nodelist=elsa-13,elsa-15 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-13,elsa-15 ${y}_${y_place}.sh >> $LOG_FILE 
				  sleep 5m
					scancel --user=gajagajago # cancel all job
					sleep 30s
        fi
				even+=1
			done
		done
	done
done

declare -i even=0
for x in transformer
do
 	for x_place in 2,2 4,2 8,2 
	do
		for y in transformer dlrm moe
		do
			for y_place in 2,2 4,2 8,2
			do
				echo "${x}_${x_place}_${y}_${y_place}" >> $LOG_FILE

        if [ $(expr $even % 2) == "0" ]
        then
          sbatch --nodelist=elsa-10,elsa-11 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-10,elsa-11 ${y}_${y_place}.sh >> $LOG_FILE
        else
          sbatch --nodelist=elsa-13,elsa-15 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-13,elsa-15 ${y}_${y_place}.sh >> $LOG_FILE 
				  sleep 5m
					scancel --user=gajagajago # cancel all job
					sleep 30s
        fi
				even+=1
			done
		done
	done
done

declare -i even=0
for x in dlrm
do
 	for x_place in 2,2 4,2 8,2 
	do
		for y in dlrm moe
		do
			for y_place in 2,2 4,2 8,2
			do
				echo "${x}_${x_place}_${y}_${y_place}" >> $LOG_FILE

        if [ $(expr $even % 2) == "0" ]
        then
          sbatch --nodelist=elsa-10,elsa-11 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-10,elsa-11 ${y}_${y_place}.sh >> $LOG_FILE
        else
          sbatch --nodelist=elsa-13,elsa-15 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-13,elsa-15 ${y}_${y_place}.sh >> $LOG_FILE 
				  sleep 5m
					scancel --user=gajagajago # cancel all job
					sleep 30s
        fi
				even+=1
			done
		done
	done
done

declare -i even=0
for x in moe
do
 	for x_place in 2,2 4,2 8,2 
	do
		for y in moe
		do
			for y_place in 2,2 4,2 8,2
			do
				echo "${x}_${x_place}_${y}_${y_place}" >> $LOG_FILE

        if [ $(expr $even % 2) == "0" ]
        then
          sbatch --nodelist=elsa-10,elsa-11 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-10,elsa-11 ${y}_${y_place}.sh >> $LOG_FILE
        else
          sbatch --nodelist=elsa-13,elsa-15 ${x}_${x_place}.sh >> $LOG_FILE
				  sbatch --nodelist=elsa-13,elsa-15 ${y}_${y_place}.sh >> $LOG_FILE 
				  sleep 5m
					scancel --user=gajagajago # cancel all job
					sleep 30s
        fi
				even+=1
			done
		done
	done
done