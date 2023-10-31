"""Checkpoint & resume logic for Slurm"""
from abc import ABCMeta, abstractmethod
from typing import TypeVar


Trainable = TypeVar("T")

class DeepShareJobCheckpointer(metaclass=ABCMeta):
    def __init__(self, checkpoint_path='', hdfs_ckpt_dir=''):
        self._checkpoint_path = checkpoint_path
        self._hdfs_ckpt_dir = hdfs_ckpt_dir
        self._latest_finished_epoch = 0

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, checkpoint_path):
        self._checkpoint_path = checkpoint_path

    def update_epoch(self, epoch):
        self._latest_finished_epoch = epoch

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self) -> Trainable:
        pass

class DummyDeepShareJobCheckpointer(DeepShareJobCheckpointer):
    def save(self) -> None:
        pass

    def load(self) -> Trainable:
        return None
