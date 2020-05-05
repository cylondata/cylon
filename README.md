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


## Build C++ Twisterx API

```bash
./build_cpp.sh
```
   

## Python Support

TwisterX provides Python APIs with Cython. 

```bash
./build_pyarrow.sh
./build_python.sh
```

### Example 

Before running the code in the base path of the cloned repo
run the following command. 

```bash
export LD_LIBRARY_PATH=$(pwd)/cpp/build/arrow/install/lib:$(pwd)/cpp/build/lib:$LD_LIBRARY_PATH
```

4. Test Python API


```bash
python3 python/test/test_pytwisterx.py
```



