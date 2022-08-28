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


On interactive node do

### Create python in venv CYLON on interactive node

NOte: once you have created CYLON there is no need to rerun this.

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
cd cylon
```


### Activities on the compute node after CYLON is available

```bash
module purge
module load gcc/9.3.0
module load spectrum-mpi/10.4.0.3-20210112
module load python/3.8-anaconda3
source ~/CYLON/bin/activate
export CC=`which gcc`
export CXX=`which g++`
export ARROW_DEFAULT_MEMORY_POOL=system
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py
pip install pytest-mpi
pip install cmake
pip install numpy
export PATH=/ccs/home/gregorvl/.local/summit/anaconda3/2020.07/3.8/bin:$PATH

cd ~/cylon
./build.sh -pyenv ~/CYLON -bpath $(pwd)/build --cpp --test --python --pytest --cmake-flags "-DMPI_C_COMPILER=$(which mpicc) -DMPI_CXX_COMPILER=$(which mpicxx)  -DCYLON_CUSTOM_MPIRUN=jsrun -DCYLON_MPIRUN_PARALLELISM_FLAG=\"-n\" -DCYLON_CUSTOM_MPIRUN_PARAMS=\"-a 1\" " -j 4
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
