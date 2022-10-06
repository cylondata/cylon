BROKEN DO NOT USE

# SUMMIT Installation guide using conda

test

## pyarrow testing 

Obtain an interactive compute node
```bash
bsub -Is -W 0:15 -nnodes 1 -P gen150_bench $SHELL
```

Load modules and create a temp env 
```bash
module purge
module load gcc/9.3.0
module load spectrum-mpi/10.4.0.3-20210112
module load python/3.8-anaconda3
source /sw/summit/python/3.8/anaconda3/2020.07-rhel8/etc/profile.d/conda.sh
conda env create -n temp
conda activate temp
```

install pyarrow 
```bash
conda install -c conda-forge pyarrow==5.0.0 -y
```

Start an interactive python shell and test the following 
```python
import pyarrow as pa 
pa.array([1, 2])
```

Delete conda env 
```bash
conda remove -n temp --all 
```

## Cylon installation 

1. Load modules 
```bash
module purge
module load gcc/9.3.0
module load spectrum-mpi/10.4.0.3-20210112
module load python/3.8-anaconda3

which python
python -V
```

2. Checkout the cylon code to `~/cylon-conda`
```bash
git clone https://github.com/cylondata/cylon.git ~/cylon-conda
cd ~/cylon-conda
```

3. Now create a conda env using the `cylon-summit.yml` file
```bash
source /sw/summit/python/3.8/anaconda3/2020.07-rhel8/etc/profile.d/conda.sh
conda env create -f summit/cylon-summit.yaml
conda activate cylon-summit
echo "python exec: $(which python)"
```

4. Update pip and install mpi4py, pytest, pytest-mpi:
```bash
pip install -U pip
CC=gcc MPICC=mpicc pip install --no-binary mpi4py install mpi4py
pip install -U pytest pytest-mpi
```

5. Submit a batch job to compile and test cylon 
```bash
bsub summit/conda.lsf
```

6. Tail the log
```bash
tail -f cylon_conda.<job_id>
```
