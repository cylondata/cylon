# Instructions 

## C++ 

- Follow [Cylon docs](https://cylondata.org/docs/) for detailed building instructions, but in summary,  
```bash
./build.sh --cpp --release
```

- Run `simple_join.cpp` example 
```bash
./build/bin/simple_join
```

- For distributed execution using MPI 
```bash
mpirun -np <procs> ./build/bin/simple_join
```

## Python 

### Build

- Activate the python virtual environment 
```bash
source <CYLON_HOME>/ENV/bin/activate 
```

- Follow [Cylon docs](https://cylondata.org/docs/) for detailed building instructions, but in summary,  
 ```bash
 ./build.sh --pyenv <CYLON_HOME>/ENV --python --release
 ```

- Export `LD_LIBRARY_PATH`
```bash
export LD_LIBRARY_PATH=<CYLON_HOME>/build/arrow/install/lib:<CYLON_HOME>/build/lib:$LD_LIBRARY_PATH
```

### Sequential Join

- Run `simple_join.py` script
```bash
python ./cpp/src/tutorial/simple_join.py
```

### Distributed Join

- For distributed execution using MPI 
```bash
mpirun -np <procs> <CYLON_HOME>/ENV/bin/python ./cpp/src/tutorial/simple_join.py
```

### Data Pre-Processing for Deep Learning with PyTorch

PyCylon pre-process the data starting from data loading and joining two tables
to formulate the features required for the data analytic carried out in PyTorch. 
PyCylon pre-process the data and releases the data as an Numpy NdArray at 
the end of the pipeline. 

#### Pre-requisites

1. Install PyTorch `pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`

 - Run sequential `simple_pytorch.py`
 
```bash
python simple_pytorch.py
```

 - Run distributed `simple_pytorch_distributed.py`
 
```bash
mpirun -n <procs> <CYLON_HOME>/ENV/bin/python simple_pytorch_distributed.py
```

`Note: procs must be set such that, 0 < procs < 5`