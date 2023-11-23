# Running Cylon on UVA CS Cluster

Arup Sarker (arupcsedu@gmail.com, djy8hg@virginia.edu)



## Install instructions

UVA CS Cluster is an HPC system offerbed by Department of Computer Science, University of Virginia.
This will use custom dependencies of the system gcc, openmpi version.

```shell
ssh your_computing_id@portal.cs.virginia.edu
git clone https://github.com/cylondata/cylon.git
cd cylon

module load gcc-11.2.0 openmpi-4.1.4

conda env create -f conda/environments/cylon.yml
conda activate cylon_dev
DIR=$HOME/anaconda3/envs/cylon_dev

export PATH=$DIR/bin:$PATH LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH PYTHONPATH=$DIR/lib/python3.9/site-packages


which python gcc g++
#---- (END) ----

python build.py --cpp --test --python --pytest

```
It will take some time to build. So, grab a coffee!!!

Let's perform a scaling operation with join. Before that, we have to install the dependencies as follow.

```shell
pip install cloudmesh-common
pip install openssl-python
```

We will slum script to run the scaling operation.

```shell
cd target/uva-cs/scripts/
sbatch scaling_job.sh
```

For more details of the dependent libraries and Slurm scripts, Please checkout the following links:

* <https://github.com/cylondata/cylon/tree/main/target/uva-cs/scripts/scaling_job.sh>
