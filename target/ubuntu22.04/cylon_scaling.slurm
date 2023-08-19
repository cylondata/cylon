#!/bin/bash
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --job-name=cylon_scaling
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

conda activate  cylon_dev


export LD_LIBRARY_PATH=$HOME/cylon/build/arrow/install/lib64:$HOME/cylon/build/glog/install/lib64:$HOME/cylon/build/lib64:$HOME/cylon/build/lib:$LD_LIBRARY_PATH

#export OMPI_MCA_pml="ucx" OMPI_MCA_osc="ucx" \
#       PATH=$DIR/bin:$DIR/libexec/gcc/x86_64-conda-linux-gnu/12.2.0:$PATH \
#       LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH \
#       PYTHONPATH=$DIR/lib/python3.10/site-packages \
#       CC=$(which mpicc) CXX=$(which mpicxx)

which python gcc g++
#---- (END) ----

python cylon_scaling.py -n 8
