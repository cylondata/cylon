---
id: docker
title: Cylon Docker Image
sidebar_label: Docker
---

Cylon has a few heavy dependencies such as OpenMPI and Apache Arrow that needs to be compiled and configured
before start writing Cylon applications. While Cylon build takes care of these things for you, optionally
following approaches can be used to quick start with the development process.

## Get started with Docker

Cylon docker images contains prebuilt cylon binaries and environment configured to start development right away.

Start by creating a volume to hold you source code.

```bash
docker volume create cylon-vol
```

Then start a Cylon container as follows with your new volume mounted at /code.

```bash
docker run -it -v cylon-vol:/code cylondata/cylon
```

Optionally, you could skip creating a volume and mount a local folder to the /code directory.

## Running Examples

The Cylon source and binaries are located at /cylon directory, and your development environment is already preloaded with everything you need to run a Cylon application locally. With the below command, you should be able to run sample applications.

```bash
cylon@d4872133cdee:~$ python3 /cylon/python/examples/dataframe/join.py
```

## Developing Cylon Applications

It's crucial to save all your work into the /code directory mounted in the above step to prevent data losses.

A cylon non-distributed application can be simple as follows. 

```python
from pycylon import DataFrame, read_csv

df1 = read_csv("file1.csv")
df1.set_index([0])

df2 = read_csv("file2.csv")
df2.set_index([0])

join = df1.merge(df2, left_on=[0], right_on=[0])
print(join)
```

Use the below command to activate the python virtual environment which comes preloaded with all the Cylon libraries.

```bash
cylon@d4872133cdee:~$ source /cylon/ENV/bin/activate
```

Assuming the above python file has been saved under /code/helloworld.py, it can be executed as follows.

```bash
(ENV) cylon@d4872133cdee:~$ python3 /code/helloworld.py
```

### Testing a distributed Cylon Application

A distributed cylon application will have an environment initialized as follows.

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

env.finalize()
```

To test a distributed application within the container, start you application with mpirun as follows. 

```bash
(ENV) cylon@d4872133cdee:~$ mpirun -np 2 python3 /code/helloworld_distributed.py
```

