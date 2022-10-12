template = """
#!/bin/bash

#BSUB -P gen150_bench
#BSUB -W {duration}
#BSUB -nnodes {nodes}
#BSUB -alloc_flags smt1
#BSUB -J cylonrun-{scaling}-{nodes}
#BSUB -o cylonrun-{scaling}-{nodes}.%J
#BSUB -e cylonrun-{scaling}-{nodes}.%J

module load python/3.7.7
source $HOME/CYLON/bin/activate

module load gcc/9.3.0

BUILD_PATH=$HOME/cylon/build
export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH


echo ################## {nodes} case
time jsrun -n $(({nodes}*42)) -c 1 python $HOME/cylon/summit/scripts/cylon_scaling.py -n {num_rows} -s {scaling}


if (({run_serial})); then
    time jsrun -n 1 -c 1 python $HOME/cylon/summit/scripts/cylon_scaling.py -n {num_rows} -s {scaling}
fi
"""

# generate weak scaling
num_rows = 50_000_000
base_time = 15
scaling = 'w'
tm = base_time
for p in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    # tm = min(base_time * p, 24 * 60)
    tm += base_time
    duration = f'{int(tm / 60)}:{tm % 60}'
    with open(f'weak_scaling/cylon_{scaling}_{p}.lsf', 'w') as f:
        f.write(template.format(duration=duration,
                                nodes=p,
                                scaling=scaling,
                                num_rows=num_rows,
                                run_serial=1 if p == 1 else 0))

# generate strong scaling
num_rows = 10_000_000_000
num_rows = num_rows - (num_rows % (256 * 42))  # adjust num_row to be a multiple of 256*42
base_time = 90
scaling = 's'
for p in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    tm = base_time
    duration = f'{int(tm / 60)}:{tm % 60}'
    with open(f'strong_scaling/cylon_{scaling}_{p}.lsf', 'w') as f:
        f.write(template.format(duration=duration,
                                nodes=p,
                                scaling=scaling,
                                num_rows=num_rows,
                                run_serial=1 if p == 1 else 0))
