## Installation

### Requirements
- Python >= 3.9
- conda
- Slurm
- CUDA
- NCCL
- Hadoop

### Setup

1. Clone 
```
git clone --recurse-submodules git@github.com:gajagajago/deepshare.git
git submodule foreach --recursive "git checkout $(git remote show origin | grep 'HEAD branch' | sed 's/.*: //')"
```

2. Set ENVs 
```
# DEEPSHARE
export DEEPSHARE_PATH=$HOME/deepshare
export DEEPSHARE_CONDA_ENV=deepshare+slurm

# SLURM
export SLURM_BUILD_PATH=/path/to/build/slurm
export PATH=$SLURM_BUILD_PATH/bin:$PATH
export SLURM_CONF_DIR=$DEEPSHARE_PATH/slurm/etc
export SLURM_CONF=$SLURM_CONF_DIR/slurm.conf

# HDFS
export JAVA_HOME=/path/to/java
export HADOOP_HOME=/path/to/hdfs
export HADOOP_BIN=$HADOOP_HOME/bin
export HADOOP_SBIN=$HADOOP_HOME/sbin
export HADOOP_DIR=/path/to/hdfs/mounted/dir
export PATH=$JAVA_HOME:$HADOOP_SBIN:$HADOOP_BIN:$HADOOP_DIR:$HADOOP_HOME:$PATH

# CONDA
export CONDA_HOME=$conda_home
```

```
# Add the following .hdfscli.cfg file on $HOME of all nodes that access the HDFS

[global]
default.alias = dev

[dev.alias]
url = http://[HDFS namenode IP]:9870
user = [HDFS user ID]
```

3. Set up base conda env
```
conda create -n deepshare+slurm python=3.9 -y
```

4. Set up packages
```
pip install .
pip install -r scheduler/requirements.txt
cd stable-baseline3 && pip install -e .
```