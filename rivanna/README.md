# Running Cylon on Rivanna

Arup Sarker (arupcsedu@gmail.com, djy8hg@virginia.edu)

## Issues on Summit

Running cylon on rivanna requires a special configuration. We have verified
that installations based on conda with virtual environment

The solution is to use the guide we provide here.

## Intsall instructions

Clone the repository

```shell
git clone https://github.com/cylondata/cylon.git
cd cylon
```

There are two ways you can build cylon on Rivanna

### Fix all dependencies for Cylon on Rivanna without the dependencies of the system gcc and openmpi version.



1. Change in yml for conda build: conda/environments/cylon.yml
Add this libraries
  - gcc
  - gxx
  - gxx_linux-64


```shell
name: cylon_rivanna
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - cmake>=3.23.1,!=3.25.0
  - arrow-cpp=9
  - pyarrow=9.0.0
  - glog
  - openmpi=4.1.3=ha1ae619_105
  - ucx>=1.12.1
  - cython>=0.29,<0.30
  - numpy
  - pandas>=1.0,<1.6.0dev0
  - fsspec>=0.6.0
  - setuptools
  # they are not needed for using pygcylon or compiling it
  - pytest
  - pytest-mpi
  - mpi4py
  - gcc
  - gxx
  - gxx_linux-64
  
 ```

Added all blue packages to overcome the dependencies on the preloaded modules.

```shell
conda env create -f conda/environments/cylon.yml
```

After creating the conda environment, there will be some warnings about OMPI_MCA_pml and OMPI_MCA_osc on ucx. To solve this problem, I have added those to the export Command. All commands are there in the slurm script.

2. Prepare Slurm script (job_slurm) to run the job. I had to add those tricks to set up the path and conda environment. Slurm scripts:

```shell
#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 10:00:00

PARENT=$HOME/.conda/envs  # parent directory of conda env
ENV=cylon_rivanna           # name of env

#---- DO NOT MODIFY THIS SECTION ----
DIR=$PARENT/$ENV
module purge
module load anaconda
source activate cylon_rivanna

export OMPI_MCA_pml="ucx" OMPI_MCA_osc="ucx" \
       PATH=$DIR/bin:$DIR/libexec/gcc/x86_64-conda-linux-gnu/12.2.0:$PATH \
       LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH \
       PYTHONPATH=$DIR/lib/python3.10/site-packages \
       CC=$(which mpicc) CXX=$(which mpicxx)

which python gcc g++
#---- (END) ----

python build.py --cpp --test --python --pytest

```
Run the Slurm script

```shell
sbatch rivana/job.slurm
```

### Build Cylon by using the loaded module of openmpi and gcc

1. Change in yml for conda build: conda/environments/cylon.yml

Comment out the openmpi package: openmpi=4.1.3=ha1ae619_105

```shell
name: cylon_rivanna
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - cmake>=3.23.1,!=3.25.0
  - arrow-cpp=9
  - pyarrow=9.0.0
  - glog
  #- openmpi=4.1.3=ha1ae619_105
  - ucx>=1.12.1
  - cython>=0.29,<0.30
  - numpy
  - pandas>=1.0,<1.6.0dev0
  - fsspec>=0.6.0
  - setuptools
  # they are not needed for using pygcylon or compiling it
  - pytest
  - pytest-mpi
  - mpi4py

```

Create virtual environment

```shell
conda env create -f conda/environments/cylon.yml
```

2. Prepare Slurm script (job_with_module.slurm) to run the job. I had to add those tricks to set up the path and conda environment. Slurm scripts:

```shell
#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p standard
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 10:00:00

PARENT=$HOME/.conda/envs  # parent directory of conda env
ENV=cylon_dev           # name of env

#---- DO NOT MODIFY THIS SECTION ----
DIR=$PARENT/$ENV
module purge
module load anaconda

module load gcc/11.2.0 openmpi/4.1.4
conda activate cylon_rivanna
export PATH=$DIR/bin:$PATH LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH PYTHONPATH=$DIR/lib/python3.8/site-packages


which python gcc g++
#---- (END) ----

python build.py --cpp --test --python --pytest

```


The two version of slurm scripts are created in this folder. Run any of them with the below command:

Run the Slurm script

```shell
sbatch rivana/job_with_module.slurm
```

 