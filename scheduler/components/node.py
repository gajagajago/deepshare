import logging

_logger = logging.getLogger(__name__)

class Node:
    def __init__(self, id: int, num_gpus: int):
        self.id = id
        self.total_gpus = self.free_gpus = num_gpus
        self.used_gpus = 0
        # List of jobs scheduled in this node
        self.scheduled_jobs = []


    def schedule_job(self, job):
        ## DEBUG
        _logger.debug(f"[Node {self.id}] Schedule {job.per_node_gpu_demand} GPUs for job {job.id}")
        ##
        self.scheduled_jobs.append(job)
        self.free_gpus -= job.per_node_gpu_demand
        self.used_gpus += job.per_node_gpu_demand


    def remove_job(self, job):
        _logger.debug(f"[Node {self.id}] remove job {job}")
        self.scheduled_jobs.remove(job)

        scheduled_jobs_str = ''
        for j in self.scheduled_jobs:
            scheduled_jobs_str += f'{j}, '
        _logger.debug(f"[Node {self.id}] scheduled jobs: {scheduled_jobs_str}")

        self.used_gpus -= job.per_node_gpu_demand
        self.free_gpus += job.per_node_gpu_demand


    def __str__(self):
        return f'Node {self.id} (used {self.used_gpus} free {self.free_gpus}): Scheduled {",".join([f"{j}" for j in self.scheduled_jobs])}'