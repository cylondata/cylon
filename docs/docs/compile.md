---
id: compile
title: Compiling
sidebar_label: Compiling
---

Cylon has C++ core, Java and Python bindings. You can compile these in three steps.

## Prerequisites

Here are the prerequisites for compiling Cylon.

1. CMake 3.16.5
2. OpenMPI 4.0.1 or higher
3. Python 3.7 or higher
4. C++ 14 or higher

## Python Environment

Because Cylon build Apache Arrow with it, we need to specify a Python environment to the build.

If you're using a virtual environment, make sure to set the virtual environment path. Or you can specify /usr as
the path if you're installing in the system path.

### Create a virtual environment

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

## OpenMPI

### From source

* We recommend using `OpenMPI 4.0.1`
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
### Installing with package manager

```bash
sudo apt install libopenmpi-dev
```

## Building Cylon From Source

We have provided a build script to make the build process easier. It is found in Cylon source root directory.
Please note that Cylon will build Apache Arrow (both `libarrow` and `pyarrow`) alongside Cylon.  

### Build C++ APIs

```bash
./build.sh -pyenv <path to your environment> -bpath <path to cmake build directory> --cpp [--release | --debug]
```

Example:

```bash
./build.sh -pyenv $HOME/cylon/ENV -bpath $HOME/cylon/build --cpp --release
```

```txt
Note: The default build mode is release 
```

### Build Python APIs

Cylon provides Python APIs with Cython.

If you're building for the first time, you can use `--all` option in build.
If you'have already built cpp and want to compile the your changes to the API,
do the following,

```bash
./build.sh -pyenv <path to your environment> -bpath <path to cmake build directory> --python
```

Note: You only need to do `--python` just once after the initial C++ build. If you develop the
Cython or Python APIs, use `--cython` flag instead.

### Updating `LD_LIBRARY_PATH`

Before running the code in the base path of the cloned repo
run the following command. Or add this to your `bashrc`.

```bash
export LD_LIBRARY_PATH=<path to cmake build dir>/arrow/install/lib:<path to cmake build dir>/lib:$LD_LIBRARY_PATH
```

### Running Tests 

You can run Cylon tests as follows. 

For C++ tests 
```bash
./build.sh -pyenv <path to your environment> -bpath <path to cmake build directory> --cpp --test
```

For Python tests 
```bash
./build.sh -pyenv <path to your environment> -bpath <path to cmake build directory> --python --pytest
```

## Building Cylon using PyArrow from Python-pip

Instead of building Cylon and Apache Arrow together, you can use [`pyarrow` distribution from`pip`](https://pypi.org/project/pyarrow/) as follows.
This will build Cylon C++ and Python APIs.

```bash
python3 -m venv <path to your env>
source <path to your env>/bin/activate 
pip install pyarrow==2.0.0

cd <cylon source dir>
./build.sh -pyenv <path to your env> -bpath <path to cmake build dir> --python_with_pyarrow  [--test | --pytest]
```

