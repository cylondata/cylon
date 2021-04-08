---
id: conda
title: Conda 
sidebar_label: Conda
---

PyCylon can be built using Conda. There are Conda packages for libcylon and pycylon. 

## Installing from Conda

The following command will install the latest version of Cylon. 

```bash
conda create -n cylon-0.4.0 -c cylondata pycylon python=3.7
conda activate cylon-0.4.0
```

## Building in a Conda environment

We need Ubuntu 16.04 or higher to build Cylon

* Ubuntu 16.04 or higher

### Install Conda & Prerequisites

First download and install Conda for your Linux distribution.

```bash
apt install wget git build-essential
```

Here are some commands used to install conda

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +x Anaconda3-2020.11-Linux-x86_64.sh
./Anaconda3-2020.11-Linux-x86_64.sh
eval "$(/root/anaconda3/bin/conda shell.bash hook)"
conda install conda-build
```
### Build Cylon

Here are the commands to build Cylon using the conda-build

```bash
git clone https://github.com/cylondata/cylon.git
cd cylon
conda-build conda/recipes/cylon/
conda-build conda/recipes/cylon/
```











