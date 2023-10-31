import subprocess
import os
import logging
import re

from scheduler.rl_training.config import NODELIST


_logger = logging.getLogger()

JOB_SCRIPT_DIR=os.path.join(os.environ["DEEPSHARE_PATH"], "slurm_examples/job_launch_scripts/isolated")
MODEL_TO_SCRIPT_DICT={
    'MobileNetV3': os.path.join(JOB_SCRIPT_DIR, "timm_{:d},{:d}.sh"),
    'GraphSage': os.path.join(JOB_SCRIPT_DIR, "graphsage_{:d},{:d}.sh"),
    'FSDP': os.path.join(JOB_SCRIPT_DIR, "fsdp_{:d},{:d}.sh"),
    'Transformer-XL': os.path.join(JOB_SCRIPT_DIR, "transformer_{:d},{:d}.sh"),
    'DLRM': os.path.join(JOB_SCRIPT_DIR, "dlrm_{:d},{:d}.sh"),
    'MoE': os.path.join(JOB_SCRIPT_DIR, "moe_{:d},{:d}.sh"),
}

# Slurm batch job launcher
# - job: target job to launch
# - placement: target nodes to launch the job to e.g., [0,1]
# - slurm_job_id_when_scheduled_before: Assigned slurm job id when scheduled before. The value is used to find ckpt and 
#                                       resume from that point.
# Return
# - slurm_job_id: Slurm job id assigned. Should be registered on job_id_to_slurm_job_id_mapper in ClusterEnv
def slurm_launch(job, placement, prev_slurm_job_id=None) -> int:
  nodelist = ",".join([NODELIST[i] for i in placement])
  num_nodes_to_schedule = len(placement)
  total_gpu_demand = job.total_gpu_demand

  script = MODEL_TO_SCRIPT_DICT[job.workload].format(total_gpu_demand, num_nodes_to_schedule)
  batch_cmd = f"sbatch --nodelist={nodelist} {script} --output={os.environ['DEEPSHARE_PATH']}/slurm_examples/out/%j.out"
  _logger.debug(f"batch_cmd: {batch_cmd}")

  # TODO: Search hdfs ckpt path if `prev_slurm_job_id` != None. Modify job launch command to make the job resume from given ckpt.
  if prev_slurm_job_id != None:
    pass

  ret = subprocess.check_output(batch_cmd, shell=True, text=True)

  try:
    slurm_job_id = int(re.search(r"\d+", ret).group(0))
  except:
    print("Error in job_launcher")

  return slurm_job_id

def slurm_preempt(slurm_job_id):
  preempt_cmd = f"scancel --signal=USR2 {slurm_job_id}"  
  _logger.debug(f"preempt_cmd: {preempt_cmd}")

  ret = subprocess.check_output(preempt_cmd, shell=True, text=True)

def slurm_kill(slurm_job_id):
  kill_cmd = f"scancel --signal=SIGKILL {slurm_job_id}"  
  _logger.debug(f"kill_cmd: {kill_cmd}")

  ret = subprocess.check_output(kill_cmd, shell=True, text=True)

