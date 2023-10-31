import numpy as np
import torch

import scheduler.rl_training.config as config
import scheduler.rl_training.obs_helper as obs_helper

import logging

_logger = logging.getLogger(__name__)


def decode_action(action):

    if action == config.INVALID:
        return []

    # Total 7 valid node assignments.
    if action == 0:
        placement = [0,1]
    elif action == 1:
        placement = [0,2]
    elif action == 2:
        placement = [0,3]
    elif action == 3:
        placement = [1,2]
    elif action == 4:
        placement = [1,3]
    elif action == 5:
        placement = [2,3]
    elif action == 6:
        placement = [0,1,2,3]

    return placement


# (1) Placement validity check: (nodes_to_schedule, job’s per-node GPUs) exists in possible placement rules
# (2) GPU availability check: for each node in nodes_to_schedule, per-node demand number of GPUs are free
# Action (i.e. nodes_to_schedule) is considered as valid iff both (1) and (2) are true.
def is_valid(candidate_jobs, obs, nodes_to_schedule):
    if nodes_to_schedule == []:
        return False, None

    num_nodes_to_schedule = len(nodes_to_schedule)
    # random.shuffle(candidate_jobs)
    for job in candidate_jobs:
        if job is not None:
            per_node_gpu_demand = int(job.total_gpu_demand / num_nodes_to_schedule)

            # (1) Placement validity check.
            # ex. [(1, 2), (2, 1)]: scheduling this job to 1 node with 2 GPUs or 2 nodes each with 1 GPU is possible.
            possible_placements = obs_helper.placement_rule(job.total_gpu_demand)
            requested_placement = (num_nodes_to_schedule, per_node_gpu_demand)
            is_placement_valid = requested_placement in possible_placements
            _logger.debug(f'=========== [ACTION] =========== (1) Job {job.id} is_placement_valid {is_placement_valid}: Total GPU demand {job.total_gpu_demand}, requested placement {requested_placement}, possible_placements {possible_placements}')

            # (2) GPU availability check.
            # For each node in nodes_to_schedule,
            # check whether adding this job's per-node GPU demand doesn't overflow the total number of GPUs in that node.
            can_allocate_per_node_gpu_demand_in_all_nodes = True
            reasons = []
            for node in nodes_to_schedule:
                num_used_gpus_in_node = np.count_nonzero(obs[node:node+1,:8], axis=1)
                if num_used_gpus_in_node + per_node_gpu_demand > 8:
                    can_allocate_per_node_gpu_demand_in_all_nodes = False
                    reasons.append(f'Among the requested nodes {nodes_to_schedule}, in node {node}, {num_used_gpus_in_node} GPUs are occupied but requested {per_node_gpu_demand} more GPUs')
                    break

            is_placement_available = can_allocate_per_node_gpu_demand_in_all_nodes
            _logger.debug(f'=========== [ACTION] =========== (2) Job {job.id} is_placement_available {is_placement_available} (reasons: {reasons})')

            if is_placement_valid and is_placement_available:
                return True, job

    return False, None

# Find possible node placement for a given job, used by LAS and SRTF inference
def find_las_srtf_placement(job, free_gpus_per_node):
    possible_placement = torch.nonzero(obs_helper.compute_possible_placement_for_single_job(job))
    _logger.debug(f"Possible placement of {job}: {possible_placement}")

    # random.shuffle(shuffled_placement)

    for n, g in reversed(possible_placement):
        placement = []

        required_nodes = n+1
        required_gpus_per_node = g+1

        # Do not allow single node placement
        if required_nodes == 1:
            break

        for node_idx, free_gpus in enumerate(free_gpus_per_node):
            if free_gpus >= required_gpus_per_node:
                placement.append(node_idx)
        _logger.debug(f"result placement: {placement}")

        if len(placement) >= required_nodes:
            return placement[:required_nodes], True

    return [], False
