#!/bin/bash

pip install cloudmesh-common -U

#PARENT=$HOME/.conda/envs  # parent directory of conda env
#ENV=cylon_dev

#---- DO NOT MODIFY THIS SECTION ----
# DIR=$PARENT/$ENV
#source activate cylon_dec

export LD_LIBRARY_PATH=$HOME/cylon/build/arrow/install/lib64:$HOME/cylon/build/glog/install/lib64:$HOME/cylon/build/lib64:$HOME/cylon/build/lib:$LD_LIBRARY_PATH

#export OMPI_MCA_pml="ucx" OMPI_MCA_osc="ucx" \
#       PATH=$DIR/bin:$DIR/libexec/gcc/x86_64-conda-linux-gnu/12.2.0:$PATH \
#       LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH \
#       PYTHONPATH=$DIR/lib/python3.10/site-packages \
#       CC=$(which mpicc) CXX=$(which mpicxx)

which python gcc g++
#---- (END) ----

python cylon_scaling.py -n 8
