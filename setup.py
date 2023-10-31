from setuptools import find_packages, setup

""" Installs DEEPSHARE packages 
Current `deepshare_packages` includes (2022.12.21)
- 'profiler', 'utils', 'profiler.scripts', 'profiler.configs', 'profiler.scripts.parser'

Run `pip install -e .` from where `setup.py` is located($DEEPSHARE_PATH) once, then `deepshare_packages` will be installed in 
editable mode. Editable mode deploys packages locally and only **links** package source code to `site-packages` 
without copy, thus additional `pip install` is not needed when the package source codes change.
"""
deepshare_packages = find_packages()
setup(
   name='deepshare',
   version='0.0',
   description='RL-based Network Contention-Aware GPU Cluster Manager for Deep Learning',
   author='',
   author_email='',
   packages=find_packages(),
   python_requires  = '>=3.9',
   install_requires=[
      # Add package only if DEEPSHARE, Slurm, or Hadoop needs it. 
      # Packages needed for misc benchmarks or scripts should be installed at execution time, 
      # with `pip install -r requirements.txt`
      'pip==23.0.1',
      'hdfs',
      'nvidia-cublas-cu11==11.10.3.66',
      'nvidia-cuda-nvrtc-cu11==11.7.99',
      'nvidia-cuda-runtime-cu11==11.7.99',
      'nvidia-cudnn-cu11==8.5.0.96',
      'psutil',
      'XlsxWriter',
      'torch==1.12.0',
   ])
