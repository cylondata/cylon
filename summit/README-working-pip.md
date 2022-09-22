# Working instructions

on a resvered node

```shell
bsub -Is -W 1:30 -nnodes 2 -P gen150_bench $SHELL
```

```shell
cd ~
git clone ...

module load python/3.7.7    
python -m venv $HOME/CYLON
source $HOME/CYLON/bin/activate

pip install pip -U
pip install pytest

export CC=`which gcc`
export CXX=`which g++`
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py
pip install -U pytest-mpi
pip install cmake
pip install numpy

module load gcc/9.3.0

cd ~/cylon
rm -rf build
BUILD_PATH=$HOME/cylon/build
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH
./build.sh -pyenv $HOME/CYLON -bpath $(pwd)/build --cpp --python --test --cmake-flags "-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx)  -DCYLON_CUSTOM_MPIRUN=jsrun -DCYLON_MPIRUN_PARALLELISM_FLAG=\"-n\" -DCYLON_CUSTOM_MPIRUN_PARAMS=\"-a 1\" " -j 4
```

This should pass all tests. If you like to run additional examples, see the next section

## Example join 

```shell
cd /ccs/home/gregorvl/cylon/python/pycylon/examples/dataframe
jsrun -n 16 python  join.py
jsrun -n 84 python  join.py
```

## Example with alias

```shell
cd ~/cylon
alias CTEST='jsrun -n 4 pytest -p no:cacheprovider --with-mpi -q'
CTEST python/pycylon/test/test_dist_rl.py  

```

# OLD

This is theold instructions, that are deprecated and should be deleted from here on.

## To install on login node

load modules

```python
module load python/3.7.7    
python -m venv $HOME/CYLON
source $HOME/CYLON/bin/activate
```
install and update pip
```python
pip install pip -U
pip install pytest
```

export gcc and cxx and install mpi4py with pytest/cmake/numpy
```python
export CC=`which gcc`
export CXX=`which g++`
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py
pip install -U pytest-mpi

pip install pytest-mpi
pip install cmake
pip install numpy
```

load gcc9 for Cylon and export build_path and build
```python
module load gcc/9.3.0
cd cylon
rm -rf build
BUILD_PATH=$HOME/cylon/build
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$LD_LIBRARY_PATH
./build.sh -pyenv $HOME/CYLON -bpath $(pwd)/build --cpp --python --cmake-flags "-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx)  -DCYLON_CUSTOM_MPIRUN=jsrun -DCYLON_MPIRUN_PARALLELISM_FLAG=\"-n\" -DCYLON_CUSTOM_MPIRUN_PARAMS=\"-a 1\" " -j 4
```

## Get an interactive node

```shell
bsub -Is -W 0:30 -nnodes 2 -P gen150_bench $SHELL
```

## To use from interactive node:

```shell
module load python/3.7.7   
module load gcc/9.3.0
```

```shell
source $HOME/CYLON/bin/activate
```

```shell
BUILD_PATH=$HOME/cylon/build
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH
```

## Example join 

```shell
cd /ccs/home/gregorvl/cylon/python/pycylon/examples/dataframe
jsrun -n 16 python  join.py
jsrun -n 84 python  join.py
```

## Example with alias

```shell
cd ~/cylon
alias CTEST='jsrun -n 4 pytest -p no:cacheprovider --with-mpi -q'
CTEST python/pycylon/test/test_dist_rl.py  

```

## THIS TEST IS NOT AVAILABLE


```shell
(CYLON) bash-4.4$ jsrun -n 2 python test_cylon.py
   _x0  _x1  _y0  _y1
0    1    2    1    2
   _x0  _x1  _y0  _y1
0    1    2    1    2
(CYLON) bash-4.4$
