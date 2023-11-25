1. Install Cloudmesh

```
pip install cloudmesh-common
pip install openssl-python

```

2. Make change in the ```cylon-experiment-setup.py ``` or ```rp-experiment-setup.py ``` for the configurations changes.

```
combination = [\
    # (1,4, 5000, "parallel", "exclusive"), # always pending
    (2,37, 1000000, "parallel", ""), 
    (4,37, 35000000, "parallel", ""), 
    (6,37, 35000000, "parallel", ""), 
    (8,37, 35000000, "parallel", ""), 
    (10,37, 35000000, "parallel", ""), 
    (12,37, 35000000, "parallel", ""), 
    (14,37, 35000000, "parallel", ""),
]
```


3. Load module and activate the python virtual environment

```
DIR=$HOME/anaconda3/envs/cylon_dev

module load gcc-11.2.0 openmpi-4.1.4
conda activate cylon_dev


export PATH=$DIR/bin:$PATH LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH PYTHONPATH=$DIR/lib/python3.9/site-packages
```
4. Run the scripts  as follows.

```bash
make clean # For cleaning 
make rp # For radical pilot
make cy # for bear metal Cylon

```