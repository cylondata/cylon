---
id: python
title: Python 
---

PyCylon is the Python binding for LibCylon (C++ Cylon). The uniqueness of PyCylon
is that it can be used as a library or a framework. PyCylon seamlessly
integrates with PyArrow and can transform between with Pandas, Numpy and Tensors. 
As a framework we support distributed relational algebra operations.

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

### Distributed Operations

Same operations can be executed in a distributed environment
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

This example shows how data can be loaded into Cylon using it's built in APIs and also using other frameworks like Pandas.
When loading from Pandas, Numpy or Apache Arrow to Cylon, there is no additional data copying overhead. When running on
a distributed environment, data can be either pre-partitioned and load based on the worker ID, or Cylon provide additional flags
to partition data if all the workers are configured to read from the same source.

```python
from pycylon import DataFrame, read_csv, CylonEnv
from pycylon.net import MPIConfig
import sys
import pandas as pd

# using cylon native reader
df = read_csv(sys.argv[1])
print(df)

# using pandas to load csv
df = DataFrame(pd.read_csv(
    "http://data.un.org/_Docs/SYB/CSV/SYB63_1_202009_Population,%20Surface%20Area%20and%20Density.csv", skiprows=6, usecols=[0, 1]))
print(df)

# loading json
df = DataFrame(pd.read_json("https://api.exchangerate-api.com/v4/latest/USD"))
print(df)

# distributed loading : run in distributed mode with MPI or UCX
env = CylonEnv(config=MPIConfig())
df = read_csv(sys.argv[1], slice=True, env=env)
print(df)
```

2. [Concat](https://github.com/cylondata/cylon/blob/master/python/examples/dataframe/concat.py)

The Concat operation is analogous to the Union operation in databases when applied across axis 0. 
If applied across axis 1, it will be similar to doing a Join.

```python
import random

import pycylon as cn
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

df1 = DataFrame([random.sample(range(10, 100), 5),
                 random.sample(range(10, 100), 5)])
df2 = DataFrame([random.sample(range(10, 100), 5),
                 random.sample(range(10, 100), 5)])
df3 = DataFrame([random.sample(range(10, 100), 10),
                 random.sample(range(10, 100), 10)])

# local unique
df4 = cn.concat(axis=0, objs=[df1, df2, df3])
print("Local concat axis0")
print(df4)

df2.rename(['00', '11'])
df3.rename(['000', '111'])
df4 = cn.concat(axis=1, objs=[df1, df2, df3])
print("Local concat axis1")
print(df4)

# distributed unique
env = CylonEnv(config=MPIConfig())

df1 = DataFrame([random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5),
                 random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5)])
df2 = DataFrame([random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5),
                 random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5)])
df3 = DataFrame([random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 10),
                 random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 10)])
print("Distributed concat axis0", env.rank)
df4 = cn.concat(axis=0, objs=[df1, df2, df3], env=env)
print(df4)

df2.rename(['00', '11'])
df3.rename(['000', '111'])
df4 = cn.concat(axis=1, objs=[df1, df2, df3], env=env)
print("Distributed concat axis1", env.rank)
print(df4)

env.finalize()
```

3. [Join](https://github.com/cylondata/cylon/blob/master/python/examples/dataframe/join.py)

Join operation can be used to merge two DataFrames across the index columns. Cylon currently support two join algorithms(Sort Join and Hash Join)
and four join types(Left, Right, Inner, Full Outer).

```python
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random

df1 = DataFrame([random.sample(range(10, 100), 50),
random.sample(range(10, 100), 50)])
df2 = DataFrame([random.sample(range(10, 100), 50),
random.sample(range(10, 100), 50)])
df2.set_index([0], inplace=True)


# local join
df3 = df1.join(other=df2, on=[0])
print("Local Join")
print(df3)

# distributed join
env = CylonEnv(config=MPIConfig())

df1 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
df2 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
df2.set_index([0], inplace=True)
print("Distributed Join")
df3 = df1.join(other=df2, on=[0], env=env)
print(df3)

env.finalize()
```

4. [Merge](https://github.com/cylondata/cylon/blob/master/python/examples/dataframe/merge.py)

Unlike the Join, Merge can be applied on non index columns. Similar to Join, Merge can be performed using two join algorithms(Sort Join and Hash Join)
and four join types(Left, Right, Inner, Full Outer).

```python
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random

df1 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])
df2 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])

# local merge
df3 = df1.merge(right=df2, on=[0])
print("Local Merge")
print(df3)

# distributed join
env = CylonEnv(config=MPIConfig())

df1 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
                 random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
df2 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
                 random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
print("Distributed Merge")
df3 = df1.merge(right=df2, on=[0], env=env)
print(df3)

env.finalize()
```

5. [Sort](https://github.com/cylondata/cylon/blob/master/python/examples/dataframe/sort.py)

Sort operation can be used to re-arrange the rows of a DataFrame based on one or more columns. If two(or more) columns are specified,
sort will be first done on the first column and then rows having similar values in the first column will be sorted based on the second column.

```python
import random

from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

df1 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])

# local sort
df3 = df1.sort_values(by=[0])
print("Local Sort")
print(df3)

# distributed sort
env = CylonEnv(config=MPIConfig())

df1 = DataFrame([random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5),
                 random.sample(range(10 * env.rank, 15 * (env.rank + 1)), 5)])
print("Distributed Sort", env.rank)
df3 = df1.sort_values(by=[0], env=env)
print(df3)

# distributed sort
print("Distributed Sort with sort options", env.rank)
bins = env.world_size * 2
df3 = df1.sort_values(by=[0], num_bins=bins, num_samples=bins, env=env)
print(df3)

env.finalize()
```

6. [Group By](https://github.com/cylondata/cylon/blob/master/python/examples/dataframe/groupby.py)

```python
from pycylon import DataFrame

df1 = DataFrame([[0, 0, 1, 1], [1, 10, 1, 5], [10, 20, 30, 40]])

df3 = df1.groupby(by=0).agg({
    "1": "sum",
    "2": "min"
})
print(df3)

df4 = df1.groupby(by=0).min()
print(df4)

df5 = df1.groupby(by=[0, 1]).max()
print(df5)
```

Group BY works similar to GROUP BY operator in databases. This should be coupled with an aggregate operation such as min, max, std, etc.

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
