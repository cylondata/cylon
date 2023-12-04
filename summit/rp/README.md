1. Install Cloudmesh

```
pip install cloudmesh-common
pip install openssl-python
python3 -m pip install urllib3==1.26.6
```

2. Make change in the ```cylon-experiment-setup.py ``` or ```cylon-experiment-setup.py ``` for the configurations changes.

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
module load gcc/9.2.0 openmpi/3.1.6 cmake/3.23.3 python/3.7.7 
source /path_to_virtual_environment/cylon_rp_venv/bin/activate
```
4. Run the scripts  as follows.

```bash
make clean # For cleaning 
make rp # For radical pilot
make cy # for bear metal Cylon

```