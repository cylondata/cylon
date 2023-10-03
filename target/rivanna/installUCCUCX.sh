#! /bin/sh


PWD=`pwd`
BUILD_PATH=$PWD/build

module load gcc/9.2.0 openmpi/3.1.6 python/3.7.7 cmake/3.23.3

python -m venv $PWD/CYLON-ENV

source $PWD/CYLON-ENV/bin/activate

# pip install -r $PWD/requirements.txt


rm -rf build

export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH

time ./build.sh -j$(nproc) -pyenv $PWD/CYLON-ENV -bpath $PWD/build --cpp --python --cython --test --cmake-flags "-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx) -DCYLON_UCX=1 -DUCX_INSTALL_PREFIX=/scratch/qad5gv/ucx/install -DCYLON_UCC=1 -DUCC_INSTALL_PREFIX=/scratch/qad5gv/ucc/install -DCYLON_USE_REDIS=1 -DREDIS_INSTALL_PREFIX=/scratch/qad5gv/redis_install"

