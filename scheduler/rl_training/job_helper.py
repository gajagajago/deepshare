import logging
import json

from scheduler.rl_training.config import MODELS


_logger = logging.getLogger(__name__)

def get_sn(job):
    job_to_sn_dict = get_job_to_sn_dict()
    job_key = f'{job.workload}-{job.total_gpu_demand}'
    return job_to_sn_dict[job_key]


# number of different jobs,
# i.e. total number of possible model x demand pairs
def get_job_to_sn_dict():
    valid_models = MODELS
    valid_demands = [2, 4, 8]
    valid_jobs = []

    for m in valid_models:
        for d in valid_demands:
            valid_jobs.append(f'{m}-{d}')

    # _logger.debug(f'valid_jobs {valid_jobs}')

    job_sn = 1
    # job serial number : job (f'{model}-{demand}') dict
    job_to_sn_dict = {}
    for job in valid_jobs:
        job_to_sn_dict[job] = job_sn
        job_sn += 1

    # _logger.debug(f'job_to_sn_dict {job_to_sn_dict}')

    return job_to_sn_dict


def get_sn_to_job_dict():
    job_to_sn_dict = get_job_to_sn_dict()
    sn_to_job_dict = {}
    for job_key, sn in job_to_sn_dict.items():
        sn_to_job_dict[sn] = job_key

    # _logger.debug(f'sn_to_job_dict {sn_to_job_dict}')

    return sn_to_job_dict

# Create jobs from trace
def parse_trace(json_trace):
    with open(json_trace) as f:
        jobs = json.load(f)
        job_list = []
        for id, props in jobs.items():
            job_dict = {}
            job_dict["id"] = int(id)
            job_dict['model'] = props['model']
            job_dict['gpu_demand'] = props['gpu_demand']
            job_dict['total_samples_to_train'] = props['total_samples_to_train']
            job_list.append(job_dict)

    return job_list