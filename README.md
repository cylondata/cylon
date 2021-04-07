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

# First example of Cylon

We can use Conda to install and run PyCylon. 

```bash
conda create -n cylon-0.4.0 -c cylondata pycylon python=3.7
conda activate cylon-0.4.0
```

Now lets run our first Cylon application

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

Now lets run a parallel version of this program.

```python

```



