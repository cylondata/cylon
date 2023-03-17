# Running Cylon on Rivanna

Arup Sarker (arupcsedu@gmail.com, djy8hg@virginia.edu)



## Intsall instructions

Rivanna is an HPC system offerbed by University of Virginia.
There are two ways you can build cylon on Rivanna.


### 1. Custom gcc conda install

This will use custom dependencies of the system gcc, openmpi version and run slurm script.


```shell
git clone https://github.com/cylondata/cylon.git
cd cylon
conda env create -f conda/environments/cylon_rivanna_1.yml
sbatch rivanna/job_cylon_rivanna_1.slurm
```

For more details of the dependent libraries and Slurm scripts, Please checkout the following links:

* <https://github.com/cylondata/cylon/tree/main/conda/environments/cylon_rivanna_1.yml>
* <https://github.com/cylondata/cylon/tree/main/rivanna/job_cylon_rivanna_1.slurm>

### 2. Module based conda install.

This will build Cylon by using the loaded module of openmpi and gcc.

Create virtual environment

```shell
git clone https://github.com/cylondata/cylon.git
cd cylon
conda env create -f conda/environments/cylon_rivanna_2.yml
sbatch rivanna/job_cylon_rivanna_2.slurm
```

For more details of the dependent libraries and Slurm scripts, Please checkout below links:

<https://github.com/cylondata/cylon/tree/main/conda/environments/cylon_rivanna_2.yml>
<https://github.com/cylondata/cylon/tree/main/rivanna/job_cylon_rivanna_2.slurm>
