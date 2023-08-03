rm -rf build
mkdir build
python -m venv CYLON
source $PWD/build/CYLON/bin/activate
pip install pip -U
export CC=`which gcc`
export CXX=`which g++`
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py
pip install -r target/ubuntu22.04/requirements.txt
BUILD_PATH=$PWD/build
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH

time ./build.sh -pyenv $PWD/cy-rp-env -bpath $(pwd)/build --cpp --python_with_pyarrow --cython --test --cmake-flags "-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx)"



1. Install Cloudmesh

```
pip install cloudmesh-common
```

2. Run the scripts in set of **compute nodes** as follows.

```bash
#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 10:00:00

PARENT=$HOME/.conda/envs  # parent directory of conda env
ENV=cylon_rivanna_1         # name of env

#---- DO NOT MODIFY THIS SECTION ----
DIR=$PARENT/$ENV
module purge
module load anaconda
source activate cylon_rivanna_1

export OMPI_MCA_pml="ucx" OMPI_MCA_osc="ucx" \
       PATH=$DIR/bin:$DIR/libexec/gcc/x86_64-conda-linux-gnu/12.2.0:$PATH \
       LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH \
       PYTHONPATH=$DIR/lib/python3.10/site-packages \
       CC=$(which mpicc) CXX=$(which mpicxx)

which python gcc g++
#---- (END) ----

python cylon_scaling.py -n 8
```