---
id: conda
title: Cylon Conda 
sidebar_label: Conda
---

PyCylon can be built using Conda. There are Conda packages for libcylon and pycylon.

## Installing from Conda

The following command will install the latest version of Cylon.

```bash
conda create -n cylon-0.4.0 -c cylondata pycylon python=3.7
conda activate cylon-0.4.0
```

Now you can run an example to see if everything is working fine.

```python
from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

df1 = DataFrame([[1, 2, 3], [2, 3, 4]])
df2 = DataFrame([[1, 1, 1], [2, 3, 4]])
df3 = df1.merge(right=df2, on=[0, 1])
print(df3)
```

## Building in a Conda environment

Now lets try to build Cylon in a Conda environment.

* Ubuntu 16.04 or higher

### Install Conda & Prerequisites

First download and install Conda for your Linux distribution.

```bash
sudo apt update && sudo apt upgrade
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install -y --no-install-recommends --no-install-suggests libssl-dev curl wget vim git build-essential python3.7-dev python3.7 maven libnuma-dev libc-dev python3-venv openmpi-bin libopenmpi-dev python3-pip python3-dev
```

Here are some commands used to install conda. Note this is an example and you can choose your own version of Conda.

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +x Anaconda3-2020.11-Linux-x86_64.sh
./Anaconda3-2020.11-Linux-x86_64.sh
```

After installing conda we need to activate the conda environment. 

```python
eval "$(~/anaconda3/bin/conda shell.bash hook)"
```

### Build Cylon

Here are the commands to build Cylon using the conda-build. These commands will build the Cylon and PyCylon packages.
We need conda-build package to build Cylon.

```bash
git clone https://github.com/cylondata/cylon.git
cd cylon

conda create --name build_env python=3.7
conda activate build_env
conda install conda-build

conda-build conda/recipes/cylon/
conda-build conda/recipes/pycylon/
```

Now you can install these packages into your conda environment. 

```bash
conda install --use-local cylon
conda install --use-local pycylon
```

After that you can use the package.

```bash
conda create -n cylon-0.4.0 -c cylondata pycylon python=3.7
conda activate cylon-0.4.0
```