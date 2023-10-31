
"""Defines handler class that all DeepShare jobs to have to be run in slurm backend."""

import utils.random_port_generator
import utils.hdfs
from utils.retry import retry
from utils.checkpoint import DeepShareJobCheckpointer
from utils.profiler import DeepShareProfiler

import signal
import os
import subprocess
import logging
import time

SLURM_USR_SIGNAL = signal.SIGUSR2


class DeepShareSlurmHandler:
    def __init__(self, checkpointer: DeepShareJobCheckpointer):
        self.checkpointer = checkpointer
        self.profiler = None

        self.job_id = os.environ["SLURM_JOB_ID"]

        signal.signal(signal.SIGCONT, self.bypass)
        signal.signal(signal.SIGTERM, self.bypass)
        signal.signal(SLURM_USR_SIGNAL, self.preempt)

        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)

        # Distributed job attributes
        self._master_addr = self._master_port = None


    def print_root(self, s):
        if os.environ['SLURM_PROCID'] == '0':
            self._logger.info(f'[rank {os.environ["SLURM_PROCID"]}] {s}')


    def print_with_rank(self, s):
        self._logger.info(f'[rank {os.environ["SLURM_PROCID"]}] {s}')


    @property
    def master_addr(self):
        if self._master_addr != None:
            return self._master_addr

        return self._get_master_addr()


    @property
    def master_port(self):
        if self._master_port != None:
            return self._master_port

        return self._get_master_port()


    # Set the latest finished epoch to use in checkpointing
    def update_epoch(self, epoch):
        self.checkpointer.update_epoch(epoch)


    """ Returns trainable if checkpoint exists on HDFS """
    def load_job(self, *args, **kwargs):
        # we should check assigned GPU memory is 0
        self._check_gpu_empty()

        # Load trainable object from ckpt path if exists on HDFS
        # Else, return None
        return self.checkpointer.load(*args, **kwargs)


    def _check_gpu_empty(self):
        pass


    def bypass(self, signum=0, frame=0):
        print(f"[DeepShareSlurmHandler] bypass -- signal{signum} received")


    def preempt(self, signum=0, frame=0):
        print(f"[DeepShareSlurmHandler] preempt -- signal{signum} received")
        print("[DeepShareSlurmHandler] Requesting checkpoint")
        self.save_checkpoint()


    def save_checkpoint(self):
        self.checkpointer.save()

        # Since Slurmd does not guarantee that job is requeued
        # after its checkpointing logic is perfectly done,
        # the job requeues itself via <scontrol requeue job_id> command.
        # after checkpointing, in order to guarantee this dependency.
        # print("[DeepShareSlurmHandler] Requesting requeue")
        # subprocess.check_call(["scontrol", "requeue", self.job_id], timeout=60)
        exit(0)

    """ Return host address of Slurm root process """
    # TODO: Clean code
    def _get_master_addr(self):
        data = subprocess.check_output(f"scontrol show jobid -dd {self.job_id} | grep BatchHost", shell=True, text=True)
        hostname = data.split("BatchHost=")[1].strip('\n')
        self._master_addr = subprocess.check_output(f"cat /etc/hosts | grep {hostname}", shell=True, text=True).split()[0]

        return self._master_addr

    """ Return host free port of Slurm root process
    The root process selects a host free port and stores it in an HDFS file, and non-root processes poll HDFS for the file (producer-consumer).
    """
    def _get_master_port(self):
        # TODO: Make below variables as ENV variables, if Fairy decides to use HDFS as main message passing medium
        hdfs_job_addr_path = '/job_addr'
        local_job_addr_path = os.path.join(os.environ['HADOOP_DIR'], 'job_addr')

        hdfs_job_port_file = os.path.join(hdfs_job_addr_path, f'{self.job_id}_port')
        local_job_port_file = os.path.join(local_job_addr_path, f'{self.job_id}_port')

        if os.environ['SLURM_PROCID'] == '0':
            self._master_port = utils.random_port_generator.main((int)(self.job_id))

            # Save master port to local path
            with open(local_job_port_file, 'w+') as f:
                f.write(f'{self._master_port}')

            # Copy the local job port file to HDFS
            utils.hdfs.upload(local_job_port_file, hdfs_job_addr_path)
        else:
            # Polling while root finishes upload
            while not utils.hdfs.exists(hdfs_job_port_file):
                self.print_with_rank('Waiting for the free port to be broadcasted as an HDFS file')
                time.sleep(1)

            # Copy the HDFS job port file to the local path
            utils.hdfs.download(hdfs_job_port_file, local_job_addr_path)

            assert os.path.exists(local_job_port_file)

            # TODO: hdfs download 1) creates file(if not exists), and 2) writes to the file.
            # If multiple workers try to download to the same path, only one worker that first finishes 1)
            # will enter 2), while others proceed. Thus, some workers may read the file in incomplete state.
            # We currently assert (as makeshift) the file to be not empty, but an alternative mechanism that
            # can guarantee the file completeness is needed.
            @retry(retry_exception=(AssertionError,))
            def _assert_not_empty(path):
                assert os.path.getsize(path) != 0

            _assert_not_empty(local_job_port_file)

            with open(local_job_port_file, 'r') as f:
                self._master_port = (int)(f.read())

        return self._master_port


    """ Write to a file """
    def file_log(self, file_path, s, mode='w'):
        with open(file_path, 'w' if mode == 'w' else 'a') as f:
            f.write(s)
            f.flush()

    """ Poll file until expected symbol is written """
    def file_poll(self, file_path, symbol=None, sleep=1):
        cnt = 0
        # TODO: When to delete symbol/file?
        while True:
            if os.path.exists(file_path):
                if symbol != None:
                    with open(file_path, 'r') as f:
                        if str(symbol) in f.read().strip():
                            break
                else:
                    break
            cnt += 1
            self.print_with_rank(f'Polling on file {file_path} (cnt: {cnt})')
            time.sleep(sleep)


    """ Install progress & resource profiler
    [GPU/CPU/Progress] profiler is installed if `log_[gpu/cpu/progress]` argument is True.
    """
    def install_profiler(self, profile_path: str = './', profile_gpu = True, profile_cpu = True, profile_progress = True, profile_iteration = 1):
        # if os.environ['SLURM_PROCID'] != '0':
        #     return

        self.profiler = DeepShareProfiler(profile_path=profile_path, profile_iteration=profile_iteration)

        # Install GPU profiler
        if profile_gpu:
            self.profiler.install_gpu_profiler()

        # Install CPU profiler
        cpu_ids = []
        if profile_cpu:
            for r in subprocess.check_output(f"scontrol show jobid -dd {self.job_id} | grep CPU_IDs", shell=True, text=True).split("CPU_IDs=")[1].split(' ')[0].split(','):
                if len(r.split('-')) > 1:
                    a, b = r.split('-')
                    for i in range(int(a), int(b)+1):
                        cpu_ids.append(i)
                else:
                    cpu_ids.append(int(r))
            self.profiler.cpu_ids = cpu_ids

        # Install Progres profiler
        if profile_progress:
            # TODO: Install if `profile_progress` is True. Currently, progress profiler is always installed
            pass


    # TODO: Implement destructor in issue #73
    def __del__(self):
        # TODO: Remove addr related file in local
        pass

        # TODO: Remove addr related file in HDFS
        if os.environ['SLURM_PROCID'] == '0':
            pass