# TwisterX
Optimized dataflow operations


## CMake Installation

### Install CMake 3.16.5 (Optional)

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz
tar zxf cmake-3.16.5.tar.gz
./bootstrap --system-curl
make 
sudo make install
```

## Python Environment

If you're using a virtual environment, make sure to set the virtual environment path. 

Or you can specify /usr as the path if you're installing in the system path. 

Create a virtual environment

```
cd  /home/<username>/build/twisterx
python3 -m venv ENV
```

Here after we assume your Python ENV path is,

```bash
 /home/<username>/build/twisterx/ENV
```

```txt
Note: User must install Pyarrow with the TwisterX build to use TwisterX APIs.
Do not use a prior installed pyarrow in your python environment. 
Uninstall it before running the setup.
```

## Build C++, Python TwisterX APIs

```bash
./build.sh -pyenv ${PYTHON_ENV_PATH} -bpath ${TWIXTER_BUILD_DIR} --all
```

Example:

```bash
./build.sh -pyenv /home/<username>/build/twisterx/ENV -bpath /home/<username>/build/twisterx/build --all
```

or 

```bash
./build.sh -pyenv /usr -bpath /home/<username>/build/twisterx/build --all
```

## Build C++ TwisterX API

```bash
./build.sh -bpath /home/<username>/build/twisterx/build --cpp
```

If you want to build each module separately make sure you build in the following order

```bash
/build.sh -bpath /home/vibhatha/build/twisterx/build --cpp
/build.sh -pyenv /home/vibhatha/build/twisterx/ENV -bpath /home/vibhatha/build/twisterx/build --pyarrow
/build.sh -pyenv /home/vibhatha/build/twisterx/ENV -bpath /home/vibhatha/build/twisterx/build --python
```

```txt
Note: The default build mode is debug
```

If you want to change the build modes, do the following

### For Release Mode

```bash
./build.sh -bpath /home/<username>/build/twisterx/build --cpp --release
```

### For Debug Mode (optional)

```bash
./build.sh -bpath /home/<username>/build/twisterx/build --cpp --debug
```   

## Python Support

TwisterX provides Python APIs with Cython. 

If you're building for the first time, you can use `--all` option in build. 
If you'have already built cpp and want to compile the your changes to the API,
do the following,

```bash
./build.sh -pyenv /home/<username>/build/twisterx/ENV -bpath /home/<username>/build/twisterx/build --python
```

### Example 

Before running the code in the base path of the cloned repo
run the following command. Or add this to your `bashrc`. 

```bash
export LD_LIBRARY_PATH=/home/<username>/twisterx/build/arrow/install/lib:/home/<username>/twisterx/build/lib:$LD_LIBRARY_PATH
```

4. Test Python API


```bash
python3 python/test/test_pytwisterx.py
```

## Samples in Google Colab (Experimental)

1. PyTwisterX Install [![PyTwisterX Install Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/vibhatha/345a4992fea18dbf27b4b61b14313b24/twisterx-install.ipynb)
2. PyTwisterX Table Demo [![PyTwisterX Table Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/vibhatha/52f0fd336be1bd9436f5050873e4aa54/pytwisterx-table-demo.ipynb)
3. PyTwisterX Pytorch Mnist Demo [![PyTwisterX Table Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/vibhatha/0e50ae349e2811c597184033a9052080/pytwisterx-pytorch-demo.ipynb)

