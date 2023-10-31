import os
import psutil
import time
import xlsxwriter
from multiprocessing import Process, Queue
import logging

import torch

from utils.retry import retry
from utils.average_meter import AverageMeter


def enforce_root(func):
    def wrapper(*args, **kwargs):
        if os.environ['SLURM_PROCID'] == '0':
            return func(*args, **kwargs)
    return wrapper


""" DeepShare Profiler module
- Progress profiler: Monitors batch processing time and estimated epoch time (batch time * batches per epoch).
- GPU profiler: Extension of PyTorch profiler. Can be installed by setting `profile_gpu` argument to True when installing
                general profiler in DeepShareSlurmHandler.
- CPU profiler: Monitors utilization(%) of job's allocated CPUs. Can be installed by setting `profile_cpu` argument to True
                when installing general profiler in DeepShareSlurmHandler. Profiles CPU utilization every `cpu_profile_interval`
                seconds. `start/stop_cpu_profile()` should be called explicitly by the job.
"""
class DeepShareProfiler:
    def __init__(self, profile_path: str = './', batches_per_epoch: int = None, profile_iteration: int = 0,
                cpu_profile_interval: float = 0.1, cpu_ids: list = []):

        self.profile_path = profile_path

        # Progress profiler
        self.profiled_steps = 0
        self.profiled_samples = AverageMeter()
        self.profiled_seconds = AverageMeter()

        # GPU profiler
        self.gpu_prof = None
        self.wait = self.warmup = self.repeat = 1
        self.active = profile_iteration

        # CPU profiler
        self.cpu_profile_interval = cpu_profile_interval
        self.cpu_ids = cpu_ids
        self.system_cpu_log = Queue() # Multiprocessing queue
        self.cpu_profile_path = os.path.join(self.profile_path, f'cpu.xlsx')
        self.cpu_monitor_proc = Process(target=self._monitor_cpu, args=())

        # Logger
        self._logger = logging.getLogger(__name__)

    def step(self, samples=0, bt=0):
        self.profiled_steps += 1
        self.profiled_samples.update(samples)
        self.profiled_seconds.update(bt)

        if self.gpu_prof != None:
            # Notify step boundary to PyTorch profiler
            self.gpu_prof.step()

        if self.profiled_steps == (self.wait + self.warmup + self.active) * self.repeat:

            self._logger.info(f'End of profile steps ({self.active})')

            if self.gpu_prof != None:
                self.stop_gpu_profile()

            # TODO: Separate CPU profiler stopping condition
            if len(self.cpu_ids) > 0:
                self.stop_cpu_profile()

            # TODO: Separate Progress profiler stopping condition
            self.stop_progress_profile()


    def install_gpu_profiler(self):
        self.gpu_prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=self.wait, warmup=self.warmup, active=self.active, repeat=self.repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profile_path),
            record_shapes=True, # set True to log network payload size
            profile_memory=True,
            with_stack=False)


    """ API to start profiling of GPUs allocated to the job """
    def start_gpu_profile(self):
        if self.gpu_prof != None:
            self.gpu_prof.start()


    """ API to stop profiling of GPUs allocated to the job """
    def stop_gpu_profile(self):
        self.gpu_prof.stop()


    """ Log system-wide CPU utilization(%) every `cpu_profile_interval` to `system_cpu_log` """
    def _monitor_cpu(self):
        while True:
            system_util = psutil.cpu_percent(interval=self.cpu_profile_interval, percpu=True)
            ts = int(time.time() * 10**6)

            op = {'ts': ts, 'util': system_util}
            self.system_cpu_log.put(op)

            time.sleep(self.cpu_profile_interval)


    """ API to start profiling of CPUs allocated to the job
    It is strongly recommended to allow only one worker to profile CPUs.
    """
    @enforce_root
    def start_cpu_profile(self):
        assert len(self.cpu_ids) > 0
        self.start_cpu_profile_caller_pid = os.getpid()
        self.cpu_monitor_proc.start()


    """ API to stop profiling of CPUs allocated to the job
    Only the process that called `start_cpu_profile` can stop the profiling
    """
    @enforce_root
    def stop_cpu_profile(self):
        # Check caller validity and wipe out caller info if valid
        assert self.start_cpu_profile_caller_pid == os.getpid()
        self.start_cpu_profile_caller_pid = None

        # Send SIGKILL to `cpu_monitor_proc` and wait for termination
        @retry(retry_exception=(AssertionError,))
        def _assert_profile_terminated():
            self.cpu_monitor_proc.kill()
            assert not self.cpu_monitor_proc.is_alive()
        _assert_profile_terminated()

        # Parse CPU log to `cpu_profile_path`
        keys = ['name', 'cat', 'ts', 'dur(us)', 'utilization(%)']
        dur = int(self.cpu_profile_interval * 10**6)
        cpu_cnt = len(self.cpu_ids)

        workbook = xlsxwriter.Workbook(self.cpu_profile_path)
        worksheet = workbook.add_worksheet()

        # Write Excel header with `keys`
        for j in range(len(keys)):
            worksheet.write(0,j,keys[j])

        # Iterate over items in `system_cpu_log` queue and write Excel rows.
        # Each queue item is separated into `cpu_cnt` Excel rows and written sequentially.
        for q in range(self.system_cpu_log.qsize()):
            data = self.system_cpu_log.get()
            for i in range(cpu_cnt):
                op = {'name': f'cpu {i}', 'cat': 'cpu_op', 'ts': data['ts'], 'dur(us)': dur, 'utilization(%)': data['util'][self.cpu_ids[i]]}
                for j in range(len(keys)):
                    excel_row_idx = q * cpu_cnt + i + 1
                    worksheet.write(excel_row_idx, j, op[keys[j]])

        workbook.close()

    @enforce_root
    def stop_progress_profile(self):
        progress_f = open(os.path.join(self.profile_path, 'progress.txt'), 'w+')
        progress_f.write(f'Avg. samples/s: {self.profiled_samples.val / self.profiled_seconds.val}\n')
        progress_f.flush()
        progress_f.close()