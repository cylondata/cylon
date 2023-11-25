# Running Cylon on Rivanna

Arup Sarker (arupcsedu@gmail.com, djy8hg@virginia.edu)



## Install instructions

Rivanna is an HPC system offerbed by University of Virginia.
This will use custom dependencies of the system gcc, openmpi version.

```shell

git clone https://github.com/cylondata/cylon.git
cd cylon

module load gcc/9.2.0 openmpi/3.1.6 python/3.7.7 cmake/3.23.3

python -m venv $PWD/cy-rp-env

source $PWD/cy-rp-env/bin/activate


pip install pip -U
pip install pytest

export CC=`which gcc`
export CXX=`which g++`
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py
pip install -U pytest-mpi
pip install numpy
pip install pyarrow==9.0.0


rm -rf build
BUILD_PATH=$PWD/build
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH

./build.sh -pyenv $PWD/cy-rp-env -bpath $(pwd)/build --cpp --python_with_pyarrow --cython --test --cmake-flags "-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx)"

```
It will take some time to build. So, grab a coffee!!!

Let's perform a scaling operation with join. Before that, we have to install the dependencies as follow.

```shell
pip install cloudmesh-common
pip install openssl-python
python3 -m pip install urllib3==1.26.6
```

We will use a slurm script to run the scaling operation.

```shell
sbatch rivanna/scripts/scaling_job.slurm
```

For more details of the dependent libraries and Slurm scripts, Please checkout the following links:

* <https://github.com/cylondata/cylon/tree/main/rivanna/scripts/scaling_job.slurm>
