import utils.hdfs
from utils.checkpoint import DeepShareJobCheckpointer

import subprocess
import os
import torch
import logging

_logger = logging.getLogger(__name__)


"""
Simplified version of the checkpoint saver in the original timm codebase.
"""
def resume_checkpoint(model, hdfs_ckpt_dir, checkpoint_path, optimizer=None):
    # Resume from the local checkpoint
    resume_epoch = None

    job_id = checkpoint_path.split("/")[-1]

    # Check if checkpoint exists on HDFS
    if not utils.hdfs.exists(f'{hdfs_ckpt_dir}/{job_id}'):
        _logger.info('Checkpoint does not exist on HDFS.')
        return resume_epoch # Must be `None`

    # Copy the HDFS checkpoint to the local directory
    utils.hdfs.download(f'{hdfs_ckpt_dir}/{job_id}', f'{checkpoint_path}')
    _logger.info('Loaded HDFS checkpoint')

    # TODO: Resolve completeness problem in issue #60
    # Resume from the local checkpoint
    if os.path.isfile(checkpoint_path):
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            _logger.info('Restoring model parameter...')

            if optimizer is not None and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                _logger.info('Restoring optimizer state from checkpoint...')

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start from the next epoch, old checkpoints incremented before save

                _logger.info('Restoring version information...')

        else:
            model.load_state_dict(checkpoint)

        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))

        raise FileNotFoundError()


class SimpleDeepShareCheckpointer(DeepShareJobCheckpointer):
    def setup(
            self,
            model,
            optimizer):

        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer

    def _save(self):

        epoch = self._latest_finished_epoch
        save_state = {
            'epoch': epoch,
            'arch': type(self.model).__name__.lower(),
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'version': 2,  # version < 2 increments epoch before save
        }

        # Local checkpointing is done using the `torch.save()`
        torch.save(save_state, self.checkpoint_path)
        _logger.info(f'Saved local checkpoint {self.checkpoint_path} at epoch {epoch}')

        # Copy the local model checkpoint to HDFS
        utils.hdfs.upload(self.checkpoint_path, self._hdfs_ckpt_dir)
        _logger.info(f'Saved HDFS checkpoint at epoch {epoch}')


    def save(self):
        if os.environ['SLURM_PROCID'] != '0':
            return

        # Delete the previous local checkpoint file first
        if os.path.exists(self.checkpoint_path):
            try:
                _logger.info(f'Removing old checkpoint: {self.checkpoint_path}')
                os.remove(self.checkpoint_path)
            except Exception as e:
                _logger.error('Exception {e} while removing {self.checkpoint_path}')

        # Save the checkpoint both as a local file and a HDFS file
        self._save()


    def load(self, model, optimizer=None):
        resume_epoch = resume_checkpoint(model, self._hdfs_ckpt_dir, self.checkpoint_path, optimizer)
        return resume_epoch
