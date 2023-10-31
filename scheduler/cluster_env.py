import logging
import pandas as pd
import numpy as np
import torch
import os
import time
from datetime import datetime
date_time_ymd = datetime.now().strftime("%Y-%m-%d")

from scheduler.components.node import Node
from scheduler.components.job import RLSched_Job
from scheduler.components.job_queue import JobQueue
from utils.average_meter import AverageMeter
from utils.job_launcher import slurm_launch, slurm_preempt, slurm_kill
import scheduler.rl_training.job_helper as job_helper
import scheduler.rl_training.obs_helper as obs_helper
import scheduler.rl_training.reward_helper as reward_helper
import scheduler.rl_training.action_helper as action_helper

from gym import Env
from gym import spaces


_logger = logging.getLogger()

class ClusterEnv(Env):
    def __init__(self, nr_node=4, gpus_per_node=8, round_dur=300, 
                rl_algo='PPO', coeff_cont=0.8, coeff_util=0.2, 
                total_jobsets=1, episodes_per_jobset=1, 
                trace=f'{os.environ["DEEPSHARE_PATH"]}/scheduler/traces/train', 
                ckpt_dir=f'{os.environ["DEEPSHARE_PATH"]}/scheduler/checkpoints/',
                isolated_thp_path=f'{os.environ["DEEPSHARE_PATH"]}/scheduler/jobs/isolated_thp.csv', 
                shared_thp_path=f'{os.environ["DEEPSHARE_PATH"]}/scheduler/jobs/shared_thp_ratio.csv'):

        super(ClusterEnv, self).__init__()

        # RL training-related
        self.rl_algo = rl_algo
        self.agg_episode_reward = 0
        # Each rollout always only contains full episodes (from beginning to terminal).
        # The number of episodes in the rollout >= 1.
        # In our default case using PPO: rollout collection = 2048 steps, episode len = 256 steps
        self.num_jobsets = 0
        self.num_episodes = 0
        self.num_timesteps = 0
        self.kth_rollout_loop = 0
        self.total_jobsets = total_jobsets
        self.episodes_per_jobset = episodes_per_jobset
        self.round_dur = round_dur
        self.episode_len = 0
        # end of episode signal
        self.dones = np.zeros(1)
        # episode reward ('r') and length ('l')
        self.infos = [{'episode': {'r': 0, 'l': 0}}] # List[Dict[str, Any]]
        self.metadata = None
        self.policy = None
        self.num_timesteps = 0
        self.kth_rollout_loop = 0
        self.coeff_cont=coeff_cont
        self.coeff_util=coeff_util
        self.ckpt_dir = ckpt_dir


        # total number of possible model x demand pairs:
        types_of_jobs = 6 * 3

        # Define an observation and action space (must be gym.spaces objects)
        self.observation_space = spaces.Box(low=0, high=types_of_jobs, shape=(4, 16), dtype=np.uint8)
        # Naive
        # self.action_space = spaces.Box(low=config.EMPTY, high=types_of_jobs, shape=(4, 8), dtype=np.uint8)
        # represent job scheduling as picking on among valid model x demand x node assignment options

        # Valid: 0-6, config.INVALID: 7
        self.action_space = spaces.Discrete(8)

        # Cluster-related
        self.trace = trace
        self.nodes = [Node(i, gpus_per_node) for i in range(0, nr_node)]
        self.total_num_gpus = nr_node * gpus_per_node
        self.num_used_gpus = 0
        self.nr_round = 0

        # Preempt jobs with shared_thp_ratio < self.preempt_shared_thp_ratio at each step
        self.preempt_shared_thp_ratio = 0.6

        self.job_queue = JobQueue()
        self.scheduled_job_list = JobQueue()
        self.scheduled_job_shared_thp_ratio = {} # { job_id: ratio }
        self.latest_scheduled_job_shared_thp_ratio_sum = 0.0
        self.candidate_jobs = None
        self.exitted_job_cnt = AverageMeter()

        self.isolated_thp = pd.read_csv(isolated_thp_path, header=0, index_col=0)
        self.shared_thp_ratio = pd.read_csv(shared_thp_path, header=0, index_col=0)

        self.system_thp = AverageMeter()
        self.jct = AverageMeter()
        self.util = AverageMeter()
        self.action_valid_ratio = AverageMeter()

        # Scheduling related. Set at start()
        self.scheduler = "rl" # <"rl"/"las"/"srtf">
        self.model = None # if scheduler = "rl", PPO loaded model
                        # else, None
        self.train = True # If True, do training. Else, do scheduling (either simulated or real)
        self.simulate = False # If True, do scheduling without Slurm launch
        self.job_id_to_slurm_job_id_mapper = None # Initialized at start(), if train && !simulate
                                                # After key lookup, value is registerd/updated. Update happens when preempted job
                                                # gets rescheduled. Entry deleted on job truncation or exit.


    def queue_jobs_from_trace(self, trace):
        job_list = job_helper.parse_trace(trace)
        for json_job in job_list:
            job = RLSched_Job(json_job['id'], json_job['model'], json_job['gpu_demand'], json_job['total_samples_to_train'])
            self.job_queue.push(job)

    def query_isolated_thp(self, job, col='isol_thp'):
        row = f'{job.workload},{job.total_gpu_demand},{job.num_nodes_to_schedule}'
        job.isolated_throughput = self.isolated_thp.at[row, col]

    def setup_learn(self, policy, episode_trace):
        self.policy = policy
        self.queue_jobs_from_trace(episode_trace)
        # total_timesteps is not used
        self.policy._setup_learn(total_timesteps=-1)


    ###
    # (1) Scheduler entry point - both for training and scheduling.
    def start(self, train=True, simulate=False, model=None, scheduler="rl"):
        self.scheduler = scheduler

        if not train:
            self.train = False
            self.simulate = simulate
            self.queue_jobs_from_trace(self.trace)
            # Plugin ckpt RL policy for scheduling
            self.model = model

            # Init slurm job id mapper for real cluster
            if not simulate:
                self.job_id_to_slurm_job_id_mapper = {}

            # Execute job scheduling until the end of the trace
            while len(self.job_queue) > 0 or len(self.scheduled_job_list) > 0:
                self.round()
        else:
            while self.num_jobsets < self.total_jobsets:
                self.num_episodes = 0
                while self.num_episodes < self.episodes_per_jobset:
                    # Load the job trace (e.g. episode0) from the train trace dir
                    # $DEEPSHARE_PATH/scheduler/traces/train/0/episode{self.num_episodes}
                    episode_trace = f'{self.trace}/jobset{self.num_jobsets}/episode{self.num_episodes}'
                    # Set the policy for training
                    self.setup_learn(model, episode_trace)
                    _logger.info(f'JOBSET {self.num_jobsets}/{self.total_jobsets-1} EPISODE {self.num_episodes}/{self.episodes_per_jobset-1} started ({episode_trace}) (cont:util {self.coeff_cont}:{self.coeff_util})')


                    # Execute job scheduling until the end of the trace
                    while len(self.job_queue) > 0 or len(self.scheduled_job_list) > 0:
                        self.round()

                    # Set the episode end signal.
                    self.episode_len = self.num_timesteps
                    avg_episode_reward = self.agg_episode_reward/self.episode_len
                    self.infos[0]['episode']['r'] = avg_episode_reward
                    self.infos[0]['episode']['l'] = self.episode_len
                    self.dones = np.ones(1)
                    _logger.info(f'JOBSET {self.num_jobsets}/{self.total_jobsets-1} EPISODE {self.num_episodes}/{self.episodes_per_jobset-1} finished (episode reward: {avg_episode_reward} episode len: {self.episode_len}) {self} (cont:util {self.coeff_cont}:{self.coeff_util})')
                    self.policy.save(f'{self.ckpt_dir}/rl-cont{self.coeff_cont}-util{self.coeff_util}-jobset{self.num_jobsets}-episode{self.num_episodes}-ckpt')

                    # One episode finished.
                    self.num_episodes += 1
                # One jobset finished.
                self.num_jobsets += 1
        
        return self.system_thp.avg, self.jct.avg, self.util.avg

    ###
    # (2) RL Training: rollout collection (env.step()) + model update

    '''
    One env step implemented in two parts:
    - Part 1. policy.compute_actions() computes actions based on the latest previous obs
    - Part 2. policy.add_new_obs_and trajectory() sends a trajectory related to the latest previous obs
      and the new obs.
    '''
    def on_policy_env_step(self):
        job = self.job_queue.peek()
        if job == None:
            return

        # (1) Compute action based on the previous obs (prev_obs)
        actions, values, log_probs, prev_obs = self.policy.compute_actions()
        placement = action_helper.decode_action(actions) # Use model, demand when needed
        valid, job_to_schedule = action_helper.is_valid(self.candidate_jobs, prev_obs[0], placement)
        if valid:
            candidate_jobs_ = ''
            for j in self.candidate_jobs:
                candidate_jobs_ += f'{j}'
            _logger.debug(f'[{self.rl_algo}] Episode {self.num_episodes} step {self.num_timesteps} : scheduling job {job_to_schedule} among the candidate jobs {candidate_jobs_}')
            self.action_valid_ratio.update(1)
            job = job_to_schedule
            self.job_queue.remove(job_to_schedule)

            job.per_node_gpu_demand = int(job.total_gpu_demand / len(placement))
            job.num_nodes_to_schedule = len(placement)

            _logger.debug(f"[VALID] Job {job} Placement {placement}")

            # Init newly scheduled job shared thp ratio
            self.scheduled_job_shared_thp_ratio[job.id] = 1.0

            # Schedule the job to the designated nodes (i.e. allocate GPUs)
            self.schedule(job, placement)

            # Update throughput of all scheduled jobs after scheduling this job
            self.update_shared_thp()

            preempt_jobs = self.preempt()
            if len(preempt_jobs) > 0:
                self.update_shared_thp()

        else:
            self.action_valid_ratio.update(0)
            self.update_shared_thp() # Required! Since jobs could have been truncated/finished after the last round, update shared thp to use for this round

            preempt_jobs = self.preempt()
            if len(preempt_jobs) > 0:
                self.update_shared_thp()

        # (2) Add a trajectory to the rollout buffer.
        # - obs: observation obtained as a result of applying the computed action
        # - rewards: how good the action was
        new_obs_array, self.candidate_jobs = obs_helper.compute_obs(self.nodes, self.job_queue)
        new_obs = torch.unsqueeze(torch.as_tensor(new_obs_array, device='cpu'), 0)
        rewards = reward_helper.compute_rewards(self.scheduled_job_shared_thp_ratio.values(), self.util.history, self.coeff_cont, self.coeff_util)
        is_full = self.policy.add_to_rollout_buffer(new_obs, actions, rewards, values, log_probs, self.dones, self.infos)
        self.agg_episode_reward += rewards

        self.num_timesteps += 1

        # New Obs: a new obs collected, as a result of the previously computed action
        # Reward: how good the previously computed action was
        # Action: a new action computed based on the previous obs
        _logger.debug(f'[{self.rl_algo}] (cont:util {self.coeff_cont}:{self.coeff_util}) Episode {self.num_episodes} Step {self.num_timesteps} Rollout Buffer Full {is_full} ({self.kth_rollout_loop}th rc loop): \nNew Obs {new_obs} Reward {self.system_thp.val} Action {actions}')

        if is_full:
            _logger.debug(f'[{self.rl_algo}] (cont:util {self.coeff_cont}:{self.coeff_util}) Episode {self.num_episodes} Step {self.num_timesteps} Rollout Buffer Full {is_full}: Model Update')
            # Do model update
            self.policy.train()
            # One policy.learn loop finished
            self.kth_rollout_loop += 1
            # init for the next rollout collection
            self.policy.init_rollout_collection()


    '''
    Resets the environment to an initial state and returns an initial observation.

    Each call of `reset()` should yield an environment suitable for
    a new episode, independent of previous episodes.

    Returns:
        observation (object): the initial observation.
    '''
    def reset(self):
        # Reset internal states of the environment (if any)
        # Produce initial observation
        initial_obs, self.candidate_jobs = obs_helper.compute_obs(self.nodes, self.job_queue)
        self.num_timesteps = 0
        self.agg_episode_reward = 0
        self.episode_len = 0
        self.dones = np.zeros(1)
        self.infos = [{'episode': {'r': 0, 'l': 0}}]
        _logger.info(f'JOBSET {self.num_jobsets} EPISODE env reset done: an environment for a new episode {self.num_episodes} is set')
        return initial_obs


    '''
    Renders the environment.
    Nothing to do.
    '''
    def render(self, mode='human'):
        pass


    '''
    Perform any necessary cleanup.
    Environments will automatically close() themselves
    when garbage collected or the program exits.
    Nothing to do.
    '''
    def close (self):
        pass


    ###
    # (3) Scheduling and removing of jobs

    def schedule(self, job, placement):
        for i in placement:
            for node in self.nodes:
                if node.id == i:
                    node.schedule_job(job)
                    self.num_used_gpus += job.per_node_gpu_demand
        
        # Query the job's isolated thp under current placement
        self.query_isolated_thp(job)

        # Add job to scheduled job list and scheduled job cs
        self.scheduled_job_list.push(job)


    # Check preemption condition for scheduled jobs, and returns a list of preempted jobs
    def preempt(self):
        preempt_jobs = JobQueue()

        for id, value in self.scheduled_job_shared_thp_ratio.items():
            if value <= self.preempt_shared_thp_ratio:
                _, job = self.scheduled_job_list.find_by_id(id)

                _logger.debug(f"Preempt {job} (shared_thp_ratio: {value})")
                preempt_jobs.push(job)

        for job in preempt_jobs:
            self.remove_job(job)
            job.isolated_throughput = 0 # reset
            self.job_queue.push(job)

        _logger.debug(f"Preempt_jobs: {preempt_jobs}")

        return preempt_jobs


    def remove_job(self, pj):
        _logger.debug(f"remove_job({pj})")
        for node in self.nodes:
            for job in node.scheduled_jobs:
                if pj.id == job.id:
                    node.remove_job(job)
                    self.num_used_gpus -= pj.per_node_gpu_demand
                    _logger.debug(f'Job {pj.id} with per-node demand {pj.per_node_gpu_demand} removed from node {node.id}: Total used GPUs: {self.num_used_gpus}')

        # Remove job entry for global data structures
        del self.scheduled_job_shared_thp_ratio[pj.id]
        self.scheduled_job_list.remove(pj)

        # Cleanup resource-related job properties
        pj.per_node_gpu_demand = 0
        pj.num_nodes_to_schedule = 0


    # Preempt all jobs
    def preempt_all(self):
        preempt_jobs = JobQueue()

        for id in self.scheduled_job_shared_thp_ratio.keys():
            _, job = self.scheduled_job_list.find_by_id(id)

            _logger.debug(f"Preempt {job}")
            preempt_jobs.push(job)

        for job in preempt_jobs:
            self.remove_job(job)
            self.job_queue.push(job)

        _logger.debug(f"Preempt_jobs: {preempt_jobs}")

        return preempt_jobs


    # One round of scheduling.
    def round(self):
        self.show_scheduling_status()

        # RL policy scheduling
        if not self.train:
            valid = True

            # Greedy action impl. Triggered if 1st action is invalid.
            greedy = False

            while valid:
                # Sort job queue for SRTF and LAS
                self.job_queue.sort(self.scheduler)

                if self.scheduler == "rl":
                    obs, candidate_jobs = obs_helper.compute_obs(self.nodes, self.job_queue)
                    actions = self.model.predict(obs)[0]
                    placement = action_helper.decode_action(actions)
                    valid, job_to_schedule = action_helper.is_valid(candidate_jobs, obs, placement)

                    # Greedy action impl.
                    if not valid and greedy == True:
                        job_to_schedule = None
                        free_gpus_per_node = [node.free_gpus for node in self.nodes]

                        for job_to_schedule in candidate_jobs:
                            if job_to_schedule is None:
                                continue
                            placement, valid = action_helper.find_las_srtf_placement(job_to_schedule, free_gpus_per_node)
                            if valid: 
                                break
                    # end greedy

                else: # las, srtfx
                    job_to_schedule = self.job_queue.peek()
                    if job_to_schedule == None:
                        break

                    free_gpus_per_node = [node.free_gpus for node in self.nodes]
                    placement, valid = action_helper.find_las_srtf_placement(job_to_schedule, free_gpus_per_node)

                _logger.debug(f"[Scheduling]"
                f"<Round {self.nr_round}> "
                f"Job {job_to_schedule} "
                f"Placement: {placement} "
                f"Valid: {valid}")

                if valid:
                    launch_target_job = job_to_schedule
                    _logger.debug(f"[Scheduling] launch_target_job {launch_target_job}")
                    self.job_queue.remove(job_to_schedule)

                    launch_target_job.per_node_gpu_demand = int(launch_target_job.total_gpu_demand / len(placement))
                    launch_target_job.num_nodes_to_schedule = len(placement)

                    # Schedule the job to the designated nodes (i.e. allocate GPUs)
                    self.schedule(launch_target_job, placement)

                    # Launch job to slurm
                    if not self.simulate:
                        prev_slurm_job_id = self.job_id_to_slurm_job_id_mapper.get(launch_target_job.id) # Check if preempted before, in order to check ckpt exists.
                        slurm_job_id = slurm_launch(launch_target_job, placement, prev_slurm_job_id)
                        self.job_id_to_slurm_job_id_mapper[launch_target_job.id] = slurm_job_id # register

                        _logger.debug(f"Registered {launch_target_job.id}: {slurm_job_id}")

                    self.scheduled_job_shared_thp_ratio[launch_target_job.id] = 1.0


            # end valid iterations
            if not self.train and not self.simulate:
                time.sleep(self.round_dur)

            # Update util
            self.update_util()

            # Update progress
            self.update_shared_thp()
            for j in self.scheduled_job_list:
                self.update_progress(j, j.isolated_throughput * self.scheduled_job_shared_thp_ratio[j.id] * self.round_dur)

            if self.scheduler == "rl":
                # Preempt jobs
                preempt_jobs = self.preempt()
            else:
                preempt_jobs = self.preempt_all()

            # Signal slurm for job preemption
            if not self.simulate:
                for j in preempt_jobs:
                    slurm_job_id = self.job_id_to_slurm_job_id_mapper.get(j.id)
                    slurm_preempt(slurm_job_id)

        # RL model training pass
        else:
            for j in self.scheduled_job_list:
                # Shared thp/isolated thp, i.e. throughput degradation due to contention.
                self.scheduled_job_shared_thp_ratio[j.id] = 1.0

            # Adaptation of SB3's On/OffPolicyAlgorithm for an environment that runs in real-time.
            self.on_policy_env_step()

            # Update util
            self.update_util()

            # Update job progress during the current round.
            for j in self.scheduled_job_list:
                self.update_progress(j, j.isolated_throughput * self.scheduled_job_shared_thp_ratio[j.id] * self.round_dur)

        self.nr_round += 1


    ###
    # (4) Update CS and progress after a round of schedule

    def update_shared_thp(self):
        # For all nodes
        for n in self.nodes:
            # For all jobs scheduled in this node
            for j in n.scheduled_jobs:

                # Assume no contention for single node scheduled jobs, since no network sharing
                if j.num_nodes_to_schedule == 1:
                    j.next_round_expected_trained_samples = j.isolated_throughput * self.round_dur
                    continue

                j_st = 1.0 # Initialized to isolated thp ratio

                for other in n.scheduled_jobs:
                    if other == j or other.num_nodes_to_schedule == 1:
                        continue
                    else:
                        this_job = f'{j.workload},{j.total_gpu_demand},{j.num_nodes_to_schedule}'
                        other_job = f'{other.workload},{other.total_gpu_demand},{other.num_nodes_to_schedule}'
                        shared_thp_ratio = self.shared_thp_ratio.at[this_job, other_job]
                        # TODO: max-min for safety (preventing shared_thp = 0 case)
                        safety_min = 0.35
                        j_st = max(min(j_st, shared_thp_ratio), safety_min)
                # Compute the worst case of thp under contention,
                # as this will be the bottleneck thp.
                if j_st <= self.scheduled_job_shared_thp_ratio[j.id]:
                    self.scheduled_job_shared_thp_ratio[j.id] = j_st
                    j.next_round_expected_trained_samples = j_st * self.round_dur


    def update_progress(self, job, samples):
        # Update the number of trained samples so far
        job.trained_samples.update(samples)
        job.trained_time.update(self.round_dur)
        _logger.debug('Job {}: (Trained ratio: {:.1f}%) (Trained time: {:d})'.format(job, job.trained_samples.sum/job.required_trained_samples*100, job.trained_time.sum))

        # Update system thp based on current round's thp
        self.system_thp.update(samples/(job.isolated_throughput*self.round_dur))

        if job.check_truncate_rule() or job.check_completion():
            _logger.debug(f"Remove finished(truncated) job {job}")
            self.remove_job(job)
            self.exitted_job_cnt.update(1)
            self.jct.update(job.trained_time.sum)

            if not self.train and not self.simulate:
                slurm_job_id = self.job_id_to_slurm_job_id_mapper.get(job.id)
                assert slurm_job_id != None
                slurm_kill(slurm_job_id)
                del self.job_id_to_slurm_job_id_mapper[job.id]
                _logger.debug(f"[Scheduling] {job} with slurm_job_id {slurm_job_id} exited.")


    ###
    # (5) Stat logging
    def show_scheduling_status(self):
        _logger.debug(self)
        _logger.debug(f"Job queue {self.job_queue}")
        _logger.debug(f"Scheduled job list {self.scheduled_job_list}")

        for node in self.nodes:
            _logger.debug(f"{node}")
        _logger.debug("\n")


    def update_util(self):
        # Cluster-wide used GPU ratio
        util = self.num_used_gpus/self.total_num_gpus
        self.util.update(util)


    def __str__(self):
        return f"Round #{self.nr_round} " + \
            f"Queued: {len(self.job_queue)} Scheduled: {len(self.scheduled_job_list)} Exitted: {self.exitted_job_cnt.sum} " + \
            f"Avg. System Thp: {self.system_thp.avg} Avg.JCT: {self.jct.avg} Avg.Util: {self.util.avg} "