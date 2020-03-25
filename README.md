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

0. Create a Virtual Environment for Development and Testing

Create an environment called `ENV`

```bash
python3 -m venv ENV
```

Activate the virtual environment (Do prior to development)

```bash
source ENV/bin/activate
```

Deactivate a pre-activated virtual environment

```bash
deactivate
```


1. Install CMake 3.16.5 (Optional)

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz
tar zxf cmake-3.16.5.tar.gz
./bootstrap --system-curl
make 
sudo make install
```

2. Install Pre-Requisites 

```bash
pip3 install -r requirements.txt
```

3. Install Python Library From Source For Development

```bash
cd python
make develop
```

4. Install Python Library From Source for Local System

```bash
cd python
make install
```

4. Test Python API

```bash
cd python
python3 test/test_pytwisterx.py
```



