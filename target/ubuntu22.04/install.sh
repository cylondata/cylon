#!/bin/sh
#
#  time sh target/ubuntu22.04/install.sh 
#
PWD=`pwd`
VENV=$PWD/build/CYLON
export BUILD_PATH=$PWD/build
echo "# ======================================================="
echo "# prepare build dir
echo "# ======================================================="
rm -rf build
mkdir build
echo "# ======================================================="
echo "# Install python venv CYLON
echo "# ======================================================="
python -m venv $VENV
which python
python --version
source $VENV/bin/activate
pip install pip -U
pip install -r $PWD/target/ubuntu22.04/requirements.txt
echo "# ======================================================="
echo "# Compile cylon
echo "# ======================================================="
export CC=`which gcc`
export CXX=`which g++`
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py

export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH

time ./build.sh -pyenv $VENV -bpath $PWD/build --cpp --python_with_pyarrow --cython --test --cmake-flags "-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx)"

echo "# ======================================================="
echo "# Install Completed
echo "# ======================================================="

