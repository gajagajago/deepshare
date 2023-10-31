import subprocess
import logging

_logger = logging.getLogger(__name__)


""" Upload local path to HDFS """
def upload(local_path, hdfs_path):
    hdfscli_cmd = f'hdfscli upload --alias=dev {local_path} {hdfs_path}'
    proc = subprocess.Popen(hdfscli_cmd, stdout=subprocess.PIPE, shell=True)
    proc.wait()
    _logger.info(f'Uploaded {local_path} to {hdfs_path}')


""" Download local path from HDFS """
def download(hdfs_path, local_path):
    hdfscli_cmd = f'hdfscli download --alias=dev {hdfs_path} {local_path}'
    proc = subprocess.Popen(hdfscli_cmd, stdout=subprocess.PIPE, shell=True)
    proc.wait() 
    _logger.info(f'Downloaded {local_path} from {hdfs_path}')


""" Check if path exists on HDFS """
def exists(hdfs_path):
    hdfscli_cmd = f'hdfs dfs -test -e {hdfs_path}'
    ret = subprocess.check_output(f'{hdfscli_cmd}; echo $?;', shell=True, text=True).strip('\n').strip()
    _logger.info(f'{hdfs_path} exists') if ret == '0' else _logger.info(f'{hdfs_path} not exists')
    return ret == '0'