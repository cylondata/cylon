import os
import sys
from textwrap import dedent
from cloudmesh.common.util import writefile
from cloudmesh.common.util import readfile
from cloudmesh.common.util import banner
from cloudmesh.common.console import Console

counter = 0

debug = True
debug = False

partition="bii-gpu"

partition="parallel"


# (nodes, threads, rows, partition, "exclusive")
combination = [\
    # (2,4, 5000, "parallel", "exclusive"), # always pending
    #(2,42, 3500000000, "debug", ""), 
    #(4,42, 35000000, "debug", ""), 
    #(8,42, 35000000, "debug", ""), 
    #(16,42, 3500000000, "debug", ""), 
    #(32,42, 3500000000, "debug", ""), 
    #(64,42, 35000000, "debug", ""), 
    (128,42, 35000000, "debug", ""),
    #(256,42, 35000000, "debug", ""),
]

'''
combination = []
for nodes in range(0,50):
  for threads in range(0,37):
    combination.append((nodes+1, threads+1, "parallel", "")) 
'''

total = len(combination)
jobid="-%j"
# jobid=""

f = open("submit.log", "w")
for nodes, threads, rows, partition, exclusive in combination:
  counter = counter + 1

  if exclusive == "exclusive":
      exclusive = "#SBATCH --exclusive"
      e = "e1"
  else:
      exclusive = ""
      e = "e0"
  
  cores_per_node = nodes * threads - 2

  print (cores_per_node)

  config = readfile("raptor.in.cfg")

  config = config.replace("CORES_PER_NODE", str(cores_per_node))
  config = config.replace("NO_OF_ROWS", str(rows))
  

  print (config)
  
  cfg_filename = f"raptor-{nodes}-{threads}.cfg"

  writefile(cfg_filename, config)
  
  banner(f"BSUB {nodes} {threads} {counter}/{total}")
  script=dedent(f"""
  #!/bin/bash
  
 #BSUB -P gen150_bench
 #BSUB -W 0:60
 #BSUB -nnodes {nodes}
 #BSUB -alloc_flags smt1
 #BSUB -J cylonrun-{nodes}
 #BSUB -o cylonrun-{nodes}.%J
 #BSUB -e cylonrun-{nodes}.%J

  module load gcc/9.3.0
  module load python/3.7.7
  source $HOME/CYLON/bin/activate

  module load gcc/9.3.0
  echo "..............................................................." 
  export RADICAL_LOG_LVL="DEBUG"
  export RADICAL_PROFILE="TRUE"
  export RADICAL_PILOT_DBURL="mongodb://arup:your_password@95.217.193.116:27017/arup"
  echo "..............................................................."  
  lscpu
  echo "..............................................................."
  BUILD_PATH=$HOME/project/dev/cylon/build
  export LD_LIBRARY_PATH=$BUILD_PATH/arrow/install/lib64:$BUILD_PATH/glog/install/lib64:$BUILD_PATH/lib64:$BUILD_PATH/lib:$LD_LIBRARY_PATH


  echo ################## 2 case
  time  python $HOME/project/dev/cylon/summit/rp/rp_scaling.py {cfg_filename} 
  #-n 50000000 -s w


  if ((0)); then
     time  python $HOME/project/dev/cylon/summit/rp/rp_scaling.py {cfg_filename} 
     #-n 50000000 -s w
  fi
  env > env-{nodes}.txt
  #time python rp_scaling.py {cfg_filename}
  echo "..............................................................."
  """).strip()

  print (script)
  filename = f"script-{nodes:02d}-{threads:02d}.lsf"
  writefile(filename, script)
  

  if not debug:

    r = os.system(f"bsub {filename}")
    total = nodes * threads
    if r == 0:
      msg = f"{counter} submitted: nodes={nodes:02d} threads={threads:02d} total={total}"
      Console.ok(msg)
    else:
      msg = f"{counter} failed: nodes={nodes:02d} threads={threads:02d} total={total}"
      Console.error(msg)
    f.writelines([msg, "\n"])
f.close()
