---
id: python
title: PyCylon
---

PyCylon is the Python binding for LibCylon (C++ Cylon). The uniqueness of PyCylon
is that it can be used as a library or a framework. As a library, PyCylon seamlessly
integrates with PyArrow. This brings us the capability of providing the user the
compatibility with Pandas, Numpy and Tensors. As a framework we support distributed
relational algebra operations using MPI as the distributed backend.

## Dataframe

PyCylon API is a Pandas like Dataframe API which supports fast, scalable, distributed memory, parallel operations.

### Initialize

In a Cylon programme, if you use Cylon with `MPI` backend, the distributed envrionment
must be initialized as follows;

```python
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
env = CylonEnv(config=MPIConfig())
```

```txt
Note: In the current release, Cylon only supports MPI as a distributed backend
```

### Load a Table

Using Cylon

```python
from pycylon import DataFrame, read_csv
df = read_csv('path/to/csv')
```

Using Pandas and convert to PyCylon Table

```python
from pycylon import DataFrame, read_csv
import pandas as pd
df = DataFrame(pd.read_csv("http://path/to/csv"))
```

Cylon Table can be converted to a PyArrow Table, Pandas Dataframe or a Numpy Array

```python
pyarrow_tb = cylon_tb.to_arrow()
pandas_df = cylon_tb.to_pandas()
numpy_arr = cylon_tb.to_numpy()
```

### PyCylon Operations

Local Operations

Local operations of PyCylon are backed by a high performance C++ core and 
can be simply executed as follows.

```python
from pycylon import DataFrame
df1 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])
df2 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])
df2.set_index([0], inplace=True)

df3 = df1.join(other=df2, on=[0])
print(df3)
```

Distributed Operations

Same operations can be executed ina distributed environment 
by simply passing the CylonEnv to the same function.

```python
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

env = CylonEnv(config=MPIConfig())

df1 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
                 random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
df2 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
                 random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])

df2.set_index([0], inplace=True)

df3 = df1.join(other=df2, on=[0], env=env)
print(df3)
```

## PyCylon Examples

1. [Data Loading](https://github.com/cylondata/cylon/blob/master/python/examples/dataframe/data_loading.py)
2. [Concat](https://github.com/cylondata/cylon/blob/master/python/examples/dataframe/concat.py)
3. [Join](https://github.com/cylondata/cylon/blob/master/python/examples/dataframe/join.py)
4. [Merge](https://github.com/cylondata/cylon/blob/master/python/examples/dataframe/merge.py)
5. [Sort](https://github.com/cylondata/cylon/blob/master/python/examples/dataframe/sort.py)
5. [Group By](https://github.com/cylondata/cylon/blob/master/python/examples/dataframe/groupby.py)

## Logging

PyCylon is backed by a C++ implementation to accelerate the operations. C++ implementation writes logs to the console for debugging purposes.
By default, logging from C++ is disabled in PyCylon. However, logging can be enabled as follows by setting CYLON_LOG_LEVEL environment variable.

```bash
export CYLON_LOG_LEVEL=<log_level_flag>
python python/examples/dataframe/join.py
```

| Log Level | Flag |
| --------- | ---- |
| INFO      | 0    |
| WARN      | 1    |
| ERROR     | 2    |
| FATAL     | 3    |

Additionally, this can be done programmatically as follows.

```python
from pycylon.util.logging import log_level, disable_logging


log_level(0) # set an arbitrary log level
disable_logging() # disable logging completely
```

## Python API docs

Use blow link to navigate to the PyCylon API docs.

<a href="/pydocs/frame.html">Python API docs</a>
