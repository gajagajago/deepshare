import utils.hdfs
from utils.checkpoint import DeepShareJobCheckpointer

from collections import OrderedDict
import logging
import os
import subprocess
import torch

_logger = logging.getLogger(__name__)


def unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

def get_state_dict(model, unwrap_fn=unwrap_model):
    return unwrap_fn(model).state_dict()

def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict

def load_state_dict(checkpoint_path, use_ema=True):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def resume_checkpoint(model, hdfs_ckpt_dir, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
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
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            state_dict = clean_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start from the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


"""
Simplified version of the checkpoint saver in the original timm codebase.
"""
class TIMMDeepShareJobCheckpointer(DeepShareJobCheckpointer):
    def setup(
            self,
            model,
            optimizer,
            args=None,
            model_ema=None,
            amp_scaler=None,
            unwrap_fn=unwrap_model):

        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler

        # config
        self.extension = '.pth.tar'
        self.unwrap_fn = unwrap_fn

    def _save(self, metric=None):
        epoch = self._latest_finished_epoch

        # Metadata for this checkpoint
        save_state = {
            'epoch': epoch,
            'arch': type(self.model).__name__.lower(),
            'state_dict': get_state_dict(self.model, self.unwrap_fn),
            'optimizer': self.optimizer.state_dict(),
            'version': 2,  # version < 2 increments epoch before save
        }
        if self.args is not None:
            save_state['arch'] = self.args.model
            save_state['args'] = self.args
        if self.amp_scaler is not None:
            save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(self.model_ema, self.unwrap_fn)
        if metric is not None:
            save_state['metric'] = metric

        # Local checkpointing is done using the `torch.save()`
        torch.save(save_state, self.checkpoint_path)
        _logger.info(f'Saved local checkpoint {self.checkpoint_path} at epoch {epoch}')

        # Copy the local model checkpoint to HDFS
        utils.hdfs.upload(self.checkpoint_path, self._hdfs_ckpt_dir)
        _logger.info(f'Saved HDFS checkpoint at epoch {epoch}')

    def save(self):
        # Delete the previous local checkpoint file first
        if os.path.exists(self.checkpoint_path):
            try:
                _logger.info(f'Removing old checkpoint: {self.checkpoint_path}')
                os.remove(self.checkpoint_path)
            except Exception as e:
                _logger.error('Exception {e} while removing {self.checkpoint_path}')

        # Save the checkpoint both as a local file and a HDFS file
        self._save()


    def load(self, model, optimizer=None, loss_scaler=None, log_info=True):
        resume_epoch = resume_checkpoint(model, self._hdfs_ckpt_dir, self.checkpoint_path, optimizer, loss_scaler, log_info)
        return resume_epoch