# Developing and Using of Cylon on Rivanna

Data: August 4th, 2023

# Using Cylon

Currently we recommend to use the instrctions for developers.
For using cylon we recommend to follow the same instructions, but replaceing the **git clone** command with

````
git clone https://github.com/cylondata/cylon.git
````


## Developing with Cylon

### Creating a working directory on Rivanna

First, we create a user directory in Rivanna's file system that is backed up so that we do not lose data/programs:

'''bash
cd /project/$USER
'''

Now, continue to follow the documentation on installing Cylon, specified next.

### Clone Cylon

First you must create a fork of cylon whcih is easiest done with the GUI.

Please visit with your browser

* <https://github.com/cylondata/cylon>

and cick on fork. Let us assume you have the username xyz, Then the easiseets is to create a shortcut for the git user
to follow our documentation.

```bash
exort GITUSER=xyz`
git clone https://github.com/$GITUSER/cylon.git
cd cylon
```

> Note: use the following line in case you do not want to fork
>
> ```bash
> git clone https://github.com/cylondata/cylon.git
> cd cylon
> ```

### Compile Cylon on Rivanna

The following lines are included in [target/rivanna/install.sh](https://github.com/cylondata/cylon/blob/main/target/rivanna/README.md)

Before executing it, please review it with

```bash
target/rivanna/install.sh
```

If you need to make modifications, copy the script and execute the copy.

If you are happy with the original script, please execute it with 

```bash
time ./target/rivanna/install.sh
```

The execution of the script will take some time.

```
real	61m17.789s
user	53m10.282s
sys   	6m52.742s
```

The script will look as follows

```bash
#! /bin/sh
PWD=`pwd`
BUILD_PATH=$PWD/build

module load gcc/9.2.0 openmpi/3.1.6 python/3.7.7 cmake/3.23.3
### We recommend that you use the following updated module versions:
# module load gcc/11.2.0 openmpi/4.1.4 python/3.11.1 cmake/3.23.3

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

## Acknowledgement

This documentation is based on documentation and installation improvements provided by

* Arup Sarker (arupcsedu@gmail.com, djy8hg@virginia.edu)
* Niranda
* Gregor von Laszewski (laszewski@gmail.com)
