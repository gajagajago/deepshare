import numpy as np
import torch
import scheduler.rl_training.job_helper as job_helper
import logging

_logger = logging.getLogger(__name__)


# Cluster state representation as numpy array with shape (num_nodes, gpus_per_node).
# Each slot corresponds to one possible placement, e.g. (1, 4) = single node, 4 GPUs
# Assign job serial no. to the slot where it is placed
# Modifed ver: jobs = jobs with 2, 4, 8 GPUs
def compute_obs(nodes, job_queue) -> torch.Tensor:
    cluster_state = compute_cluster_state(nodes)    # (4, 8)
    candidate_jobs = pick_jobs_with_multiple_demands(job_queue)
    # possible_placement = obs_helper.compute_possible_placement_head(job) # (4, 8)
    possible_placement = compute_possible_placement_multiple_demands(candidate_jobs) # (4, 8)
    _logger.debug(f''' =========== [OBS] =========== possible placement
    {possible_placement}
    ''')
    return torch.cat((cluster_state, possible_placement), 1), candidate_jobs


def pick_jobs_with_multiple_demands(job_queue):
    # In current job_queue,
    # greedily pick jobs with different demand, i.e. 2, 4, 8 GPU jobs
    # starting from the head of the queue.
    demand_to_job_dict = {2: None, 4: None, 8: None}
    demand_str = ''
    for job in job_queue:
        # print(f'demand_ Job {job.id} Demand {job.total_gpu_demand}')
        if demand_to_job_dict[2] is None and job.total_gpu_demand == 2:
            demand_to_job_dict[2] = job
            demand_str += f' Demand 2: Job {job.id}'
        elif demand_to_job_dict[4] is None and job.total_gpu_demand == 4:
            demand_to_job_dict[4] = job
            demand_str += f' Demand 4: Job {job.id}'
        elif demand_to_job_dict[8] is None and job.total_gpu_demand == 8:
            demand_to_job_dict[8] = job
            demand_str += f' Demand 8: Job {job.id}'

        # if the dict is full, stop iterating
        if all(v is not None for v in demand_to_job_dict.values()):
            break

    _logger.debug(f'demand_to_job_dict {demand_str}')

    return list(demand_to_job_dict.values())


def compute_cluster_state(nodes) -> np.array:
    cluster_state = np.zeros((4, 8))

    # Translate current cluster state into (num_nodes, gpus_per_node) shape array.
    for node in nodes:
        for job in node.scheduled_jobs:
            # If the job is expected to finish/truncate after the next round, exclude the job when
            # computing the cluster state
            if job.check_truncate_rule():
                continue

            # Consecutive allocation of unused GPUs in the scheduled node
            demand = 0
            for gpu_idx in range(0, node.total_gpus):
                if cluster_state[node.id, gpu_idx] == 0:
                    cluster_state[node.id, gpu_idx] = job_helper.get_sn(job)
                    demand += 1
                    if demand == job.per_node_gpu_demand:
                        break

    tensor = torch.from_numpy(cluster_state)
    _logger.debug(f'''Computing the current cluster state:
    {tensor} ({tensor.shape})''')

    return tensor


def compute_possible_placement_multiple_demands(jobs) -> np.array:
    possible_placement = np.zeros((4, 8))
    job_ids = ''
    job_sns = ''

    for job in jobs:
        if job is not None:
            job_ids += f'{job.id} '
            job_sn = job_helper.get_sn(job)
            job_sns += f'{job_sn} '
            # The given job's possible placements.
            # e.g. job with demand = 4: (1, 4), (2, 2), (4, 1)
            node_gpus_l = placement_rule(job.total_gpu_demand)
            # row: node id
            # col: number of GPUs per node
            for placement in node_gpus_l:
                num_nodes = placement[0] - 1
                per_node_gpus = placement[1] - 1
                # currently, we have only 4 nodes in the cluster
                if placement[0] > 4:
                    _logger.debug(f'Cannot place ({num_nodes}x{per_node_gpus}): node {num_nodes} does not exist')
                else:
                    possible_placement[num_nodes, per_node_gpus] = job_sn

    tensor = torch.from_numpy(possible_placement)
    _logger.debug(f'''Computing possible placements of jobs ( ID {job_ids}) ( SN {job_sns}):
    {tensor} ({tensor.shape})''')
    return tensor


def compute_possible_placement_for_single_job(job) -> np.array:
    possible_placement = np.zeros((4, 8))

    if job != None:
        job_sn = job_helper.get_sn(job)

        # The given job's possible placements.
        # e.g. job with demand = 4: (1, 4), (2, 2), (4, 1)
        node_gpus_l = placement_rule(job.total_gpu_demand)
        # row: node id
        # col: number of GPUs per node
        for placement in node_gpus_l:
            num_nodes = placement[0] - 1
            per_node_gpus = placement[1] - 1
            if placement[0] > 4:
                _logger.debug(f'Cannot place ({num_nodes}x{per_node_gpus}): node {num_nodes} does not exist')
            else:
                possible_placement[num_nodes, per_node_gpus] = job_sn

    tensor = torch.from_numpy(possible_placement)
    _logger.debug(f'''Computing possible placements of {job} (SN {job_sn}):
    {tensor} ({tensor.shape})''')
    return tensor


# (node, GPUs) placement, either isolated or shared
def placement_rule(gpu_demand):
    if gpu_demand == 2:
        return [(1, 2), (2, 1)]
    elif gpu_demand == 4:
        return [(1, 4), (2, 2), (4, 1)]
    elif gpu_demand == 8:
        return [(1, 8), (2, 4),(4, 2)]
