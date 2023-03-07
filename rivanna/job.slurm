#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 10:00:00

PARENT=$HOME/.conda/envs  # parent directory of conda env
ENV=cylon_rivanna          # name of env

#---- DO NOT MODIFY THIS SECTION ----
DIR=$PARENT/$ENV
module purge
module load anaconda
source activate cylon_rivanna

export OMPI_MCA_pml="ucx" OMPI_MCA_osc="ucx" \
       PATH=$DIR/bin:$DIR/libexec/gcc/x86_64-conda-linux-gnu/12.2.0:$PATH \
       LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH \
       PYTHONPATH=$DIR/lib/python3.10/site-packages \
       CC=$(which mpicc) CXX=$(which mpicxx)

which python gcc g++
#---- (END) ----

python build.py --cpp --test --python --pytest
