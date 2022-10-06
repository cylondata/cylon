# Running Cylon on Summit

Gregor von Laszewski (laszewski@gmail.com), and Niranda Perera

## Issues on Summit

Running cylon on summit requires a special install. We have verified
that installations based on conda and anaconda do not work as it
includes many incompatible packages that do not work on summit. Hence
we do not recommend that you even attempt using anaconda and conda.
In addition, we ran into library incompatibilities between the python
and C/C++-libraries as cylon needs very particular version management.

The solution is to use the guide we provide here.

## Intsall instructions

If you have already installed cylon, you can skip this section and go to "Using the installed version"

These instructions are to be executed on 2 reserved nodes on summit as
to not only compiles cylon, but also runs a minimal cylon tests to see if
cylon works.

> **Note:** Although the compilation could be done on a front-end node, the
> tests must be run on the worker nodes.

To give ample time we recommend the nodes be reserved for 1h 30min so
you can after the compile test it out. The documentation here is
focused on a compilation done in an interactive node. This has the
advantage that if errors occur they can be debugged
interactively. Once completed, we anticipate that a batch script is
made available for the compilation.

To obtain the interactive nodes, please use the command:

```shell
bsub -Is -W 1:30 -nnodes 2 -P gen150_bench $SHELL
```

Now on the interactive nodes, we recommend a complete new installation
of cylon in order to capture the latest code. This may require you to
remove the ~/cylon directory. The following commands document the
process:


```shell
cd ~
git clone https://github.com/cylondata/cylon.git
# git clone git@github.com:cylondata/cylon.git
cd cylon
git checkout summit
cd ~

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

The approximate time for running this is less then an hour.

## Example join 

This test is done to execute a quick running example to see if things work.

```shell
cd /ccs/home/gregorvl/cylon/python/pycylon/examples/dataframe
time jsrun -n 16 python  join.py 
time jsrun -n 84 python  join.py
```

The code with -n 16 completes in about 15s

The code with -n 84 completes in about 13s

## Using the installed version

In case you have cylon installed and want to run additional tests at a different login, 
you can use the compiled version as follows.

First, you need to obtian some nodes for some time (here 2 nodes with 1h 30min)

```shell
bsub -Is -W 1:30 -nnodes 2 -P gen150_bench $SHELL
```

Next, activate the prebuild cylon

```bash
module load python/3.7.7  
source $HOME/CYLON/bin/activate

module load gcc/9.3.0 

BUILD_PATH=$HOME/cylon/build
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH
```

Finally run an short example to see if things work

```
cd /ccs/home/gregorvl/cylon/python/pycylon/examples/dataframe
time jsrun -n 16 python  join.py
```

This code witll complete in approximately 15s.

## Example with alias

```shell
cd ~/cylon
alias CTEST='jsrun -n 4 pytest -p no:cacheprovider --with-mpi -q'
CTEST python/pycylon/test/test_dist_rl.py  

```

The approximate time for running this script is: TBD

## Example: Long running job

in the directory summit/scripts/we find a documentation and a script for 
a long-running job helping to benchmark cylon on summit. 

Run the scripts in set of **compute nodes** as follows.

```bash
module load python/3.7.7  
source $HOME/CYLON/bin/activate

module load gcc/9.3.0 

BUILD_PATH=$HOME/cylon/build
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH

cd ~/cylon/summit/scripts
time jsrun -n <parallelism> python weak_scaling.py
```

The approximate time for running it is less than 15 seconds

A script that changes the parameter of n is located in the scripts dir and can be called with 

```bash
sh ./benchmark-summit.sh 2>&1 | tee -a summit.log
```