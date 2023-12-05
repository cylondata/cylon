#!/bin/bash

##BSUB -P gen150_bench
##BSUB -W 0:45
##BSUB -nnodes 2
##BSUB -alloc_flags smt1
##BSUB -J cylonrun-w-2
##BSUB -o cylonrun-w-2.%J
##BSUB -e cylonrun-w-2.%J

#module load python/3.7.7
#source $HOME/CYLON/bin/activate

#module load gcc/9.3.0
echo "..............................................................." 
export RADICAL_LOG_LVL="DEBUG"
export RADICAL_PROFILE="TRUE"
export RADICAL_PILOT_DBURL="mongodb://You_Mongodb_url"
echo "..............................................................."  
lscpu
echo "..............................................................."
BUILD_PATH=$HOME/project/dev/cylon/build
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH


echo ################## 2 case
time  python $HOME/project/dev/cylon/summit/rp/rp_scaling.py raptor-2-42.cfg 
#-n 50000000 -s w


if ((0)); then
   time  python $HOME/project/dev/cylon/summit/rp/rp_scaling.py raptor-2-42.cfg 
   #-n 50000000 -s w
fi
#time python rp_scaling.py raptor-2-42.cfg
echo "..............................................................."
