# Running Cylon on Rivanna

Arup Sarker (arupcsedu@gmail.com, djy8hg@virginia.edu)



## Install instructions for Radical Pilot

Rivanna is an HPC system offerbed by University of Virginia.
This will use custom dependencies of the system gcc, openmpi version.
Use the same python environment "cylon_rct" for radical-pilot

```shell
module load gcc/9.2.0 openmpi/3.1.6 python/3.7.7 cmake/3.23.3
source $HOME/cylon_rct/bin/activate
pip install radical.pilot
```
For checking all dependent library version:

```shell
radical-stack
```
You need to export mongo-db url:

```shell
export RADICAL_PILOT_DBURL="mongodb://rct-tutorial:HXH7vExF7GvCeMWn@95.217.193.116:27017/rct-tutorial"
```
Setup is done. Now let's execute scaling with cylon.

```shell
cd /some_path_to/cylon/rivanna/rp-scripts
python rp_scaling.py
```

If you want to make any change in the uva resource file(/some_path_to/radical.pilot/src/radical/pilot/configs) or any other places in the radical pilot source,

```shell
git clone https://github.com/radical-cybertools/radical.pilot.git
cd radical.pilot
```
For reflecting those change, you need to upgrade radical-pilot by,

```shell
pip install . --upgrade
```

To uninstall radical pilot, execute

```shell
pip uninstall radical.pilot
```