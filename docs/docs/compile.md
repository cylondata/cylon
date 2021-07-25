---
id: compile
title: Source Compilation
sidebar_label: Source
---

Cylon has C++ core, Java and Python bindings. You can compile these in three steps.

Cylon can be build along with Arrow (Cylon will build Apache Arrow) or it can be build by pointing to an existing
Arrow installation.

This document shows how to build Cylon on Linux and Mac OS. The first section of the document shows how to install
the required dependencies on Linux (Ubuntu) and Mac OS. After required dependencies are installed, 
the compiling is similar in Linux and Mac OS.

## Prerequisites

Here are the prerequisites for compiling Cylon.

1. CMake 3.16.5
2. OpenMPI 4.0.1 or higher (You can use any other MPI version as well, we tested with OpenMPI)
3. Python 3.7 or higher
4. C++ 14 or higher

### Python Environment

We need to specify a Python environment to the build script. If you're using a virtual environment, 
make sure to set the virtual environment path. Or you can specify /usr as the path if you're installing in the system path.

#### Create a virtual environment

```
cd  $HOME/cylon
python3 -m venv ENV
source ENV/bin/activate
```

Here after we assume your Python ENV path is,

```bash
 $HOME/cylon/ENV
```

```txt
Note: User must install Pyarrow with the Cylon build to use Cylon APIs.
Do not use a prior installed pyarrow in your python environment.
Uninstall it before running the setup.
```

### Installing Dependencies Ubuntu

Cylon uses MPI for distributed execution. So we need an MPI version installed in the system. There are many implementations
of MPI standard such as MPICH and OpenMPI. We have tested Cylon with OpenMPI and you should be able to use any other MPI implementation like
MPICH as well. 

In this document we will explain how to install OpenMPI. You can use the following command to install OpenMPI on 
an Ubuntu system. If you would like to build OpenMPI with custom options, please refer to their [documentation](https://www.open-mpi.org/faq/?category=building) or you can
follow the quick tutorial at the end of the document to do so. 

```bash
sudo apt install libopenmpi-dev
```

Here are some of the other dependencies required. 

```bash
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update && apt-get install -y --no-install-recommends --no-install-suggests libssl-dev curl wget vim git build-essential python3.7-dev python3.7 maven libnuma-dev libc-dev python3-venv openmpi-bin libopenmpi-dev python3-pip python3-dev libutf8proc-dev libre2-dev
```

We need a later version of CMake. We can build cmake from source if the version in our system is less than 3.16.5.

```bash
curl -OL https://github.com/Kitware/CMake/releases/download/v3.20.1/cmake-3.20.1.tar.gz
tar -xvf cmake-3.20.1.tar.gz
cd cmake-3.20.1
./bootstrap
make
sudo make install
```

### Installing Dependencies MacOS

You would need to install XCode and install an MPI version such as OpenMPI.

```bash
brew install open-mpi
```

Once those are installed you are ready to compile Cylon on macos.

## Build Cylon & PyCylon on Linux or Mac OS

Here we will walk you through building Cylon along with Apache Arrow.

We have provided a build script to make the build process easier. It is found in Cylon source root directory.
Please note that Cylon will build Apache Arrow (both `libarrow` and `pyarrow`) alongside Cylon.  

### Build C++ APIs

```bash
./build.sh -pyenv <path to your environment> -bpath <path to cmake build directory> -ipath <path to binary install directory> --cpp [--release | --debug]
```

Example:

```bash
# make the cylon cpp library install directory
mkdir $HOME/cylon_install
./build.sh -pyenv $HOME/cylon/ENV -bpath $HOME/cylon/build -ipath $HOME/cylon_install --cpp --release
```

```txt
Note: The default build mode is release 
```

Now lets try to run an C++ example and see whether our compilation is successful.

```bash
# export the lib path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cylon_install/lib
cd $HOME/cylon_install
# this will run the join example with randomly generated data
./examples/join_example m o 40000 1 sort
```

It will generate an output like following. 

```bash
I0714 01:49:30.757613 867371 join_example.cpp:57] Sort join algorithm
I0714 01:49:30.763633 867371 join_example.cpp:79] Read tables in 363[ms]
I0714 01:49:30.840909 867371 partition_op.cpp:80] 0 Partition start: Wed Jul 14 01:49:30 2021 time: 0
I0714 01:49:30.843380 867371 all_to_all_op.cpp:72] Shuffle time: 2
I0714 01:49:30.851606 867371 split_op.cpp:152] Split time: 10 Fin time: 8 Split time: 2 Call count: 1
I0714 01:49:30.852095 867371 partition_op.cpp:80] 0 Partition start: Wed Jul 14 01:49:30 2021 time: 0
I0714 01:49:30.854277 867371 all_to_all_op.cpp:72] Shuffle time: 2
I0714 01:49:30.862313 867371 split_op.cpp:152] Split time: 10 Fin time: 8 Split time: 2 Call count: 1
I0714 01:49:30.953450 867371 join_kernel.cpp:57] Done concatenating tables, rows :  39712
I0714 01:49:30.976418 867371 join_op.cpp:46] Join time : 110
I0714 01:49:31.037182 867371 join_example.cpp:100] First table had : 40000 and Second table had : 40000, Joined has : 39712
I0714 01:49:31.038038 867371 join_example.cpp:102] Join done in 272[ms]
```

### Build Python APIs

Cylon provides Python APIs with Cython. Cylon will build, Cylon CPP, Cylon Python, Arrow CPP and Arrow Python here. In this mode
it will install the Cylon and PyCylon libraries to the Python environment using pip. We only support pip through source builds.
If you want to use an existing Cylon binary you would need to use Conda packages. 

You can use the following command to build the Python library.

```bash
./build.sh -pyenv <path to your environment> -bpath <path to cmake build directory> --python
```

Here is an example command.

```bash
./build.sh -pyenv $HOME/cylon/ENV -bpath $HOME/cylon/build -ipath $HOME/cylon_install --python
```

This command will install the PyCylon and PyArrow into the virtual environment we specified. 

#### Updating library path

Before running the code in the base path of the cloned repo you need to update the runtime library path. Linux and Mac OS uses different environment variable names.
Following are two commands to update the path on these operating systems. 

#### Linux

```bash
export LD_LIBRARY_PATH=<path to cmake build dir>/arrow/install/lib:<path to cmake build dir>/lib:$LD_LIBRARY_PATH
```

Here is an example command.
```bash
export LD_LIBRARY_PATH=$HOME/cylon/build/arrow/install/lib:$HOME/cylon/build/lib:$LD_LIBRARY_PATH
```

#### Mac OS

```bash
export DYLD_LIBRARY_PATH=<path to cmake build dir>/arrow/install/lib:<path to cmake build dir>/lib:$DYLD_LIBRARY_PATH
```

Here is an example command.
```bash
export DYLD_LIBRARY_PATH=$HOME/cylon/build/arrow/install/lib:$HOME/cylon/build/lib:$DYLD_LIBRARY_PATH
```

After this you can verify the build.

```bash
source ENV/bin/activate
```
Here is an example PyCylon programs to check whether installation is working.

```python
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

df1 = DataFrame([[1, 2, 3], [2, 3, 4]])
df2 = DataFrame([[1, 1, 1], [2, 3, 4]])
df3 = df1.merge(right=df2, on=[0, 1])
print(df3)
```
Congratulations you now have successfully installed PyCylon and Cylon.

### Running Tests 

You can run Cylon tests as follows. 

For C++ tests 
```bash
./build.sh -pyenv <path to your environment> -bpath <path to cmake build directory> --cpp --test
```

Here is an example command.

```bash
./build.sh -pyenv $HOME/cylon/ENV -bpath $HOME/cylon/build --cpp --test
```

For Python tests

```bash
./build.sh -pyenv <path to your environment> -bpath <path to cmake build directory> --python --pytest
```

Here is an example command

```bash
./build.sh -pyenv $HOME/cylon/ENV -bpath $HOME/cylon/build --python --test
```

## Building Cylon With An Existing Arrow Installation

If you already have an arrow installation and wants to use that for the build, you can do so by pointing the build to that.

### Building PyCylon

Instead of building PyCylon and Apache Arrow together, you can use [`pyarrow` distribution from`pip`](https://pypi.org/project/pyarrow/) as follows.
This will build only the Cylon C++ and Python APIs. Here we will use the arrow libraries from the PyArrow installation.

First lets create a Python environment and install PyArrow in it. 

```bash
python3 -m venv ENV
source ENV/bin/activate 
pip install pyarrow==4.0.0
```

Then we can build Cylon pointing to this pyarrow with the following command.

```bash
./build.sh -pyenv <path to your env> -bpath <path to cmake build dir> --python_with_pyarrow  [--test | --pytest]
```

Here is an example command.

```bash
cd $HOME/cylon
./build.sh -pyenv $HOME/cylon/ENV -bpath $HOME/cylon/build --python_with_pyarrow
```

After this you can run the above PyCylon examples to make sure it is working.

## Building OpenMPI From Source 

In this section we will explain how to build and install OpenMPI 4.0.1 from source. The instructions can be used to build a higher
version of OpenMPI as well.

* We recommend using `OpenMPI 4.0.1` or higher.
* Download OpenMPI 4.0.1 from [https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz](https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz)
* Extract the archive to a folder named `openmpi-4.0.1`
* Also create a directory named `build` in some location. We will use this to install OpenMPI
* Set the following environment variables

  ```text
  BUILD=<path-to-build-directory>
  OMPI_401=<path-to-openmpi-4.0.1-directory>
  PATH=$BUILD/bin:$PATH
  LD_LIBRARY_PATH=$BUILD/lib:$LD_LIBRARY_PATH
  export BUILD OMPI_401 PATH LD_LIBRARY_PATH
  ```

* The instructions to build OpenMPI depend on the platform. Therefore, we highly recommend looking into the `$OMPI_401/INSTALL` file. Platform specific build files are available in `$OMPI_401/contrib/platform` directory.
* In general, please specify `--prefix=$BUILD` and `--enable-mpi-java` as arguments to `configure` script. If Infiniband is available \(highly recommended\) specify `--with-verbs=<path-to-verbs-installation>`. Usually, the path to verbs installation is `/usr`. In summary, the following commands will build OpenMPI for a Linux system.

  ```text
  cd $OMPI_401
  ./configure --prefix=$BUILD --enable-mpi-java
  make -j 8;make install
  ```