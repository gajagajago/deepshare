#!/bin/bash

### Setup script for conda & requirements

# Only local root process manages setup
if [ "$SLURM_LOCALID" != "0" ]; then
    echo "setup.sh called from non-local root process. exit."
    exit
else
    echo "setup.sh $SLURMD_NODENAME"
fi

# Options
CONDA_ENV=$SLURM_JOBID
REQUIREMENTS=""
LINKS=""

# Parse options
while (( "$#" )); do
    case "$1" in
        -c|--conda)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                CONDA_ENV=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        -r|--requirements)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                REQUIREMENTS=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        -f|--find-links)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                LINKS=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        -h|--help)
            echo "Usage:  $0 -i <input> [options]" >&2
            echo "        -c | --conda  %  (set conda environment to ... (Default: Slurm job id))" >&2
            echo "        -r | --requirements  %  (install requirements from ... (Default: none))" >&2
            echo "        -f | --find-links  %  (install requirements with ... (Default: none))" >&2
            exit 0
            ;;
        -*|--*) # unsupported flags
            echo "Error: Unsupported flag: $1" >&2
            echo "$0 -h for help message" >&2
            exit 1
            ;;
        *)
            echo "Error: Arguments with not proper flag: $1" >&2
            echo "$0 -h for help message" >&2
            exit 1
            ;;
    esac
done
echo "===parsed command line option==="
echo " - conda: ${CONDA_ENV}"
echo " - requirements: ${REQUIREMENTS}"
echo " - links: ${LINKS}"

# Create conda env cloning from `DEEPSHARE_CONDA_ENV`
# `conda create -n` creates env only if the named env does not exist.
# conda env is created with slurm job id if CONDA_ENV option is not given.
# Assumption: conda env with the current slurm job id does not exist
. $CONDA_HOME/etc/profile.d/conda.sh
conda create -n $CONDA_ENV --clone $DEEPSHARE_CONDA_ENV
conda activate $CONDA_ENV

# Install requirements
if [ "$REQUIREMENTS" != "" ]; then
    if [ "$LINKS" != "" ]; then
	    pip install -r $REQUIREMENTS -f $LINKS
    else
        pip install -r $REQUIREMENTS
    fi
fi

conda deactivate