rm -rf build
mkdir build
time python -m venv $PWD/build/CYLON
source $PWD/build/CYLON/bin/activate
pip install pip -U
export CC=`which gcc`
export CXX=`which g++`
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py
pip install -r $(PWD)/target/ubuntu22.04/requirements.txt
BUILD_PATH=$PWD/build
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH

time ./build.sh -pyenv $PWD/cy-rp-env -bpath $(pwd)/build --cpp --python_with_pyarrow --cython --test --cmake-flags "-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx)"

