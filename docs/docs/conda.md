---
id: conda
title: Cylon Conda Binaries
sidebar_label: Conda
---

Cylon can be built and used through a Conda environment. 
There are Conda packages for Cylon C++ and Python libraries (libcylon and pycylon).

## Installing from Conda

The following command will install the latest version of Cylon.

```bash
conda create -n cylon-0.4.0 -c cylondata pycylon python=3.7
conda activate cylon-0.4.0
```

Now you can run an example to see if everything is working fine.

```python
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

df1 = DataFrame([[1, 2, 3], [2, 3, 4]])
df2 = DataFrame([[1, 1, 1], [2, 3, 4]])
df3 = df1.merge(right=df2, on=[0, 1])
print(df3)
```

## Building in a Conda environment

Now lets try to build Cylon in a Conda environment.

* Ubuntu 16.04 or higher

### Install Conda & Prerequisites

First download and install Conda for your Linux distribution.

```bash
sudo apt update && sudo apt upgrade
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install -y --no-install-recommends --no-install-suggests libssl-dev curl wget vim git build-essential python3.7-dev python3.7 maven libnuma-dev libc-dev python3-venv openmpi-bin libopenmpi-dev python3-pip python3-dev
```

Here are some commands used to install conda. Note this is an example and you can choose your own version of Conda.

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +x Anaconda3-2020.11-Linux-x86_64.sh
./Anaconda3-2020.11-Linux-x86_64.sh
```

After installing conda we need to activate the conda environment. 

```python
eval "$(~/anaconda3/bin/conda shell.bash hook)"
```

### Build Cylon

Here are the commands to build Cylon using the conda-build. These commands will build the Cylon and PyCylon packages.
We need conda-build package to build Cylon.

```bash
git clone https://github.com/cylondata/cylon.git
cd cylon

conda env create -f conda/environments/cylon.yml
conda activate cylon_dev
./build.sh --conda_cpp --conda_python
```

After that you can use PyCylon or libcylon as explained above.

Here, Built files can be found in the `$CYLON_HOME/build` 
(build directory can be specified from the command line with 
`-bpath` flag as: ./build.sh [-bpath \<build dir>] ... )

Additionally, Cylon libraries would also be installed to `$CONDA_PREFIX/lib` and
`$CONDA_PREFIX/include` directories.

## Installing and Using GCylon

GPU Cylon (gcylon) provides distributed dataframe processing on NVIDIA GPUs. 
There are two libraries for gcylon: 
* a cpp library (libgcylon) 
* a python library (pygcylon)

GCylon libraries depend on Cylon libraries and NVIDIA libraries: cudatoolkit and cudf

Since cudatoolkit and cudf libraries are rather large, we provide a separate conda environment for installing and compiling gcylon.

The easiest way to compile and run gcylon is through a conda environment. We provide a conda environment yml file. It has all dependencies listed.

### Prerequisites 
* Clone the cylon project to your machine from github if not already done.
* Make sure you have anaconda or miniconda installed. If not, please install anaconda or miniconda first. 
* Install cudatoolkit 11.0 or higher. 
* Make sure your machine is Linux and has:
  - NVIDIA driver 450.80.02+
  - A GPU with Pascal architecture or better (Compute Capability >=6.0)

### Installing Conda Packages
Go to cylon project directory on the command line.

Check your cudatoolkit installation version. You can check it with:
```
nvcc --version
```

If your cudatoolkit version is not 11.2, update the cudatoolkit version at the file:
conda/environments/gcylon.yml

Create the conda environment and install the dependencies, 
activate the conda environment:
```
conda env create -f conda/environments/gcylon.yml
conda activate gcylon_dev
```

Compile and Install Cylon cpp and python packages:
```
./build.sh --conda_cpp --conda_python
```

Compile and Install GCylon cpp and python packages:
```
./build.sh --gcylon --pygcylon
```

Checking whether pycylon and pygcylon packages are installed after the compilation:
```
conda list | grep cylon
pycylon                 "version number"
pygcylon                "version number"
```

Running the join example from gcylon examples directory:
Running with 2 mpi workers (-n 2) on the local machine:
```
mpirun -n 2 --mca opal_cuda_support 1 python python/pygcylon/examples/join.py
```

To enable ucx, add the flags "--mca pml ucx --mca osc ucx" to the mpirun command.  
To enable infiniband, add the flag "--mca btl_openib_allow_ib true" to the mpirun command.  
To run the join example with both ucx and infiniband enabled on the local machine with two mpi workers:
```
mpirun -n 2 --mca opal_cuda_support 1 --mca pml ucx --mca osc ucx --mca btl_openib_allow_ib true python python/pygcylon/examples/join.py
```

Other examples in the python/pygcylon/examples/ directory can be run similarly.


## Setting up IDEs 

In addition to use terminal, you can also use the Conda environment in your preferred IDE's. 

1. Open Cylon as a C++ project, and assign `cylon/cpp/CmakeLists.txt` as main CMake file.

2. Export `CONDA_PREFIX=<path to env>` environment variable for the IDE

3. Add a CMake build directory (ex: `$CYLON_HOME/build`)

4. Use the following CMake options
```bash
-DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX"
-DARROW_BUILD_TYPE="SYSTEM"
-DARROW_LIB_DIR="$CONDA_PREFIX/lib"
-DARROW_INCLUDE_DIR="$CONDA_PREFIX/include"
-DCYLON_PARQUET=ON # enable Cylon parquet 
-DPYCYLON_BUILD=ON # enable PyCylon 
-DCYLON_WITH_TEST=ON # run C++ tests 
```
