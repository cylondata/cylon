Run the scripts in set of **compute nodes** as follows.

```bash
module load python/3.7.7  
source $HOME/CYLON/bin/activate

module load gcc/9.3.0 

BUILD_PATH=$HOME/cylon/build
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH

jsrun -n <parallelism> python 
```