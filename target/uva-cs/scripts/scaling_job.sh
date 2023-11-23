#!/bin/bash --login

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=20
#SBATCH --time=0:60:00
#SBATCH --job-name="Cylon Scaling" 
#
#SBATCH --time=0:60:00
#SBATCH --partition=main
#SBATCH --error="%j-stderr.txt"
#SBATCH --output="%j-stdout.txt"
#
           

DIR=$HOME/anaconda3/envs/cylon_dev

module load gcc-11.2.0 openmpi-4.1.4
conda activate cylon_dev


export PATH=$DIR/bin:$PATH LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH PYTHONPATH=$DIR/lib/python3.9/site-packages


which python gcc g++

mpirun -np 20 python cylon_scaling.py -n 35000000

