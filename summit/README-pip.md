# SUMMIT Installation guide

This document describes the instalation on summit. 
The instalition is first described in detail and then an abbreviated 
instalation with a Makefile is provided.

Notation:

* `login$` denotes the login node
* `compute$` donates a compute node

## Details

Now we obtain two interactive compute nodes

```bash
login$ bsub -Is -W 0:30 -nnodes 2 -P gen150_bench $SHELL
```

After you get them you will be in a compute node


On ointeractive node do

### Activites on the login node

```bash
module purge
module load gcc/9.3.0
module load spectrum-mpi/10.4.0.3-20210112
module load python/3.8-anaconda3
```

Next check the python version 

```bash
which python
> /sw/summit/python/3.8/anaconda3/2020.07-rhel8/bin/python
python -V
> Python 3.8.3
```

Now create a venv called `CYLON` in the home directory and activate it

```bash
python -m venv ~/CYLON
source ~/CYLON/bin/activate
```

```bash
which python
> ~/CYLON/bin/python
python -V
> Python 3.8.3
```

Update pip and install pytest:

```bash
pip install pip -U
pip install pytest
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py
pip install -U pytest-mpi
```
```

Checkout the cylon code

```bash
cd ~git clone https://github.com/cylondata/cylon.git
```


### Activities on the compute node

```bash
compute$ module purge
compute$ module load gcc/9.3.0
compute$ module load spectrum-mpi/10.4.0.3-20210112
compute$ module load python/3.8-anaconda3
compute$ source ~/CYLON/bin/activate
compute$ export CC=`which gcc`
compute$ export CXX=`which g++`
compute$ export ARROW_DEFAULT_MEMORY_POOL=system
compute$ CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py
compute$ pip install pytest-mpi
compute$ pip install cmake
compute$ pip install numpy
compute$ export PATH=/ccs/home/gregorvl/.local/summit/anaconda3/2020.07/3.8/bin:$PATH

compute$ ./build.sh -pyenv ~/CYLON -bpath $(pwd)/build --cpp --test --cmake-flags "-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx)  -DCYLON_CUSTOM_MPIRUN=jsrun -DCYLON_MPIRUN_PARALLELISM_FLAG=\"-n\" " -j 8
```

The compilation will take some time. After it is completed you can conduct a test with

```bash
compute$ cd ~/cylon/build
compute$ jsrun -n 1 -c 4 -a 1 ./build/bin/join_example m n 1000 0.9
```

If everything is ok, you will see at the end of the test output

```
...
================================================
All tests passed (66 assertions in 4 test cases)
```


### Running batch scripts

Please note that the module load and the source of the CYLON venv must be at the beginning of each batsch script you want to use cylon in.
