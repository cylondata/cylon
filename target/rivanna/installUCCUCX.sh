#! /bin/sh


PWD=`pwd`
BUILD_PATH=$PWD/build

module load gcc/11.2.0  openmpi/4.1.4 python/3.11.1 cmake/3.23.3

python -m venv $PWD/CYLON-ENV

source $PWD/CYLON-ENV/bin/activate

# pip install -r $PWD/requirements.txt
pip install pip -U
pip install pytest
pip install -U pytest-mpi
pip install numpy
pip install cloudmesh-openstack

# pip install pyarrow==9.0.0

export CC=`which gcc`
export CXX=`which g++`
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py


rm -rf build

export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH

time ./build.sh -j$(nproc) -pyenv $PWD/CYLON-ENV -bpath $PWD/build -ucxpath /scratch/qad5gv/ucx-1.12.1/install -uccpath /scratch/qad5gv/ucc/install2 -redispath /scratch/qad5gv/redis_install --cpp --python --cython --pytest --cmake-flags "-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx) -DCYLON_UCX=1 -DUCX_INSTALL_PREFIX=/scratch/qad5gv/ucx-1.12.1/install -DCYLON_UCC=1 -DUCC_INSTALL_PREFIX=/scratch/qad5gv/ucc/install2 -DCYLON_USE_REDIS=1 -DREDIS_INSTALL_PREFIX=/scratch/qad5gv/redis_install"
