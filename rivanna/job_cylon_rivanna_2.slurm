#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 10:00:00

PARENT=$HOME/.conda/envs  # parent directory of conda env
ENV=cylon_rivanna_2          # name of env

#---- DO NOT MODIFY THIS SECTION ----
DIR=$PARENT/$ENV
module purge
module load anaconda

module load gcc/11.2.0 openmpi/4.1.4
conda activate cylon_rivanna_2
export PATH=$DIR/bin:$PATH LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH PYTHONPATH=$DIR/lib/python3.8/site-packages


which python gcc g++
#---- (END) ----

python build.py --cpp --test --python --pytest

