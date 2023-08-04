1. Install Cloudmesh

```
pip install cloudmesh-common
pip install openssl-python
python3 -m pip install urllib3==1.26.6
```

2. Run the scripts in set of **compute nodes** as follows.

```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --partition=bii
#SBATCH -A bii_dsc_community
#SBATCH --output=rivanna/scripts/cylogs/mpirun-96t-4n-160w-35m-%x-%j.out
#SBATCH --error=rivanna/scripts/cylogs/mpirun-96t-4n-160w-35m-%x-%j.err


module load gcc/9.2.0 openmpi/3.1.6 cmake/3.23.3 python/3.7.7

#module load gcc/11.2.0
#module load openmpi/4.1.4
#module load python/3.11.1

#source $HOME/CYLON/bin/activate
source $HOME/cylon_rp_venv/bin/activate

BUILD_PATH=$PWD/build

export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH


which python gcc g++


#srun -n 160 python $PWD/rivanna/scripts/cylon_scaling.py -n 35000000
mpirun -np 160 python rivanna/scripts/cylon_scaling.py -n 35000000

```