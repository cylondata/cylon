# Cylon

[![Build Status](https://travis-ci.org/cylondata/cylon.svg?branch=master)](https://travis-ci.org/cylondata/cylon)
[![License](http://img.shields.io/:license-Apache%202-blue.svg)](https://github.com/cylondata/cylon/blob/master/LICENSE)

Cylon is a fast, scalable distributed memory data parallel library
for processing structured data. Cylon implements a set of relational operators to process data.
While ”Core  Cylon” is implemented using system level C/C++, multiple language interfaces
(Python  and  Java ) are provided to seamlessly integrate with existing applications, enabling
both data and AI/ML engineers to invoke data processing operators in a familiar programming language.
By default it works with MPI for distributing the applications.

Internally Cylon uses [Apache Arrow](https://arrow.apache.org/) to represent the data in a column format.

The documentation can be found at [https://cylondata.org](https://cylondata.org)

Email - [cylondata@googlegroups.com](mailto:cylondata@googlegroups.com)

Mailing List - [Join](https://groups.google.com/forum/#!forum/cylondata/join)

# Getting Started

We can use Conda to install PyCylon. At the moment Cylon only works on Linux Systems. The Conda binaries need Ubuntu 16.04 or higher. 

```bash
conda create -n cylon-0.4.0 -c cylondata pycylon python=3.7
conda activate cylon-0.4.0
```

Now lets run our first Cylon application inside the Conda environment. The following code creates two DataFrames and joins them. 

```python
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

df1 = DataFrame([[1, 2, 3], [2, 3, 4]])
df2 = DataFrame([[1, 1, 1], [2, 3, 4]])

# local merge
df3 = df1.merge(right=df2, on=[0, 1])
print("Local Merge")
print(df3)
```

Now lets run a parallel version of this program. Here if we create n processes (parallelism), n instances of the
program will run. They will each load two DataFrames in their memory and do a distributed join among the DataFrames.
The results will be created in the parallel processes as well. 

```python
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random

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
```

You can run the above program in the Conda environment by using the following command. It uses ```mpirun``` command with 2 parallel processes.  

```bash
mpirun -np 2 python <name of your python file>
```

# Compiling Cylon

Refer to the documentation on how to compile Cylon

[Compiling on Linux](https://cylondata.org/docs/)

# Licence

Cylon uses the Apache Lincense Version 2.0




