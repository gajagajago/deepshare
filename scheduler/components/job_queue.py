import logging
from queue import PriorityQueue
import random


_logger = logging.getLogger(__name__)

class PriorityJobQueue:

    def __init__(self):
        self.queue = PriorityQueue()

    def empty(self):
        return self.queue.empty()

    def put(self, job):
        self.queue.put(job)

    def get(self):
        return self.queue.get()

    def __str__(self):
        s = '{ \n'
        for job in self.queue.queue:
            s += '\t' + str(job) + '\n'
        s += '} \n'
        return s

    def __len__(self):
        return self.queue.qsize()


class JobQueue:
    def __init__(self):
        self.q = []
        self._current_index = 0

    def push(self, job):
        self.q.append(job)

    def pop(self, idx=0):
        return self.q.pop(idx)

    def remove(self, job):
        return self.q.remove(job)

    def peek(self):
        return self.q[0] if self.__len__() > 0 else None

    def at(self, idx):
        return self.q[idx] if self.__len__() > idx else None
    
    def find_by_id(self, id):
        for idx, job in enumerate(self.q):
            if job.id == id:
                return idx, job
        return -1, None
    
    def sort(self, scheduler="rl"):
        if scheduler == "las":
            self.q.sort(key=lambda job: job.trained_time.sum, reverse=True)
        elif scheduler == 'srtf':
            self.q.sort(key=lambda job: job.trained_samples.sum/job.required_trained_samples)

    def shuffle(self):
        random.shuffle(self.q)
        
    def __str__(self):
        s = '{ \n'
        for job in self.q:
            s += '\t' + str(job) + '\n'
        s += '} \n'
        return s

    def __len__(self):
        return len(self.q)

    def __iter__(self):
        self._current_index = 0
        return self
    
    def __next__(self):
        if self._current_index < self.__len__():
            ret = self.at(self._current_index)
            self._current_index += 1
            return ret

        raise StopIteration