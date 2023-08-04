#~ /bin/sh

PWD=`pwd`
BUILD_PATH=$PWD/build

module load gcc/9.2.0 openmpi/3.1.6 python/3.7.7 cmake/3.23.3

python -m venv $PWD/cy-rp-env

source $PWD/cy-rp-env/bin/activate

# pip install -r $PWD/requirements.txt

pip install pip -U
pip install pytest
pip install -U pytest-mpi
pip install numpy
# pip install pyarrow==9.0.0

export CC=`which gcc`
export CXX=`which g++`
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py

rm -rf build

export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH

time ./build.sh -pyenv $PWD/cy-rp-env -bpath $PWD/build --cpp --python --cython --test --cmake-flags "-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx)"
```

