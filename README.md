# TwisterX
Optimized dataflow operations

# Build instructions
   - Build any MPI of preference
   - Update LD_LIBRARY_PATH to include lib directory containing MPI libraries
   - First go to the `cpp` directory,
   - Create a directory `build`
   - `cd build`
   - `cmake -DCMAKE_BUILD_TYPE=Debug ../`
   - `make -j4`
   

## Python Support

TwisterX provides Python APIs with Pybind11. 

### Pre-requisites

1. Install CMake 3.16.5

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz
tar zxf cmake-3.16.5.tar.gz
./bootstrap --system-curl
make 
sudo make install
```

2. Install Pybind11 

```bash
pip3 install pybind11
sudo apt-get install python-pybind11
```

3. Install Python Library From Source

```bash
cd cpp
python3 setup.py install
```

4. Test Python API

```bash
python3 python/test/test_twister.py
```



