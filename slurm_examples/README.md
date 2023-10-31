# Example benchmarks

Benchmarks to test the Slurm scheduler.

## How to write benchmark script

- Slurm commands are executed by all processes involved. 
- Commands for setup & installation must be executed by at most one process per node. Else, duplicated execution of such codes can show unexpected behaviors.
- Slurm batch script is blocking. All processes process each line of command simultaneously.

1. Configure `SBATCH` options **at the beginning of the script**
2. Configure default values for options
3. Setup conda env & install packages using `setup.sh`. If other requirements are needed, refer to an example in `bert_ddp.sh`(under `#Install other requirements (optional)`)
4. Add the training launch script at the end

## Issues
1. All worker nodes must have the same copy of benchmark program(e.g. timm_ddp.py)
2. All worker nodes must have the same copy of `setup.sh`
3. All worker nodes must have the same copy of requirements file, if required.

## How to run
Submit the training script to Slurm using `sbatch` command. `sbatch` command transfers the script with the `#SBATCH` directives into a job, and places the job into the scheduler queue. Available $priority options can be found using `sinfo` command.
```
sbatch -p $priority $training_script.sh
```