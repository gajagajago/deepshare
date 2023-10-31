from utils.average_meter import AverageMeter
import scheduler.rl_training.job_helper as job_helper


class RLSched_Job:
    def __init__(self, id, workload, gpu_demand, required_trained_samples):
        ## Constant properties

        self.workload = workload
        self.total_gpu_demand = gpu_demand
        self.required_trained_samples = required_trained_samples

        ## Dynamic properties
        
        self.id = id # int # Under inference, SLURM_JOBID. 
                        # Under simulated inference, ClusterEnv.simulated_inference_job_id
                        # Under training, uniquely given from JobHelper.parse_trace

        self.num_nodes_to_schedule = 0 # Assign when job is scheduled
        self.per_node_gpu_demand = 0 # Assign when job is scheduled

        self.isolated_throughput = 0 # Assign when job is scheduled

        self.trained_samples = AverageMeter()
        self.next_round_expected_trained_samples = 0 # updated at update_shared_thp()

        self.trained_time = AverageMeter()

    # When less than 1 round of running remains
    def check_truncate_rule(self):
        return self.required_trained_samples - self.trained_samples.sum < self.next_round_expected_trained_samples

    def check_completion(self):
        return self.required_trained_samples <= self.trained_samples.sum

    def __str__(self):
        job_key = f'{self.workload}-{self.total_gpu_demand}'
        return f'[ID{self.id}][Type{job_helper.get_sn(self)}]{job_key}'