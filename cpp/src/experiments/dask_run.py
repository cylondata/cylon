import os

import dask.dataframe as dd 
from dask.distributed import Client, SSHCluster

import time 
import argparse
import math
import subprocess


parser = argparse.ArgumentParser(description='generate random data')

parser.add_argument('-s', dest='scale', type=str, help='number of rows', required=True)
parser.add_argument('-w', dest='world', type=int, help='number of rows', required=True, nargs='+')
parser.add_argument('-r', dest='rows', type=int, help='number of rows', required=True, nargs='+')
parser.add_argument('-i', dest='it', type=int, help='number of rows', default=1)


args = parser.parse_args()
args = vars(args)
# print(args, flush=True)

scale = args['scale']
world = args['world']
rows = args['rows']
it = args['it']

"ssh <IP> dask-worker --interface enp175s0f0 --scheduler-file sched.json --local-directory /scratch/dnperera/dask/ --interface enp175s0f0 --memory-limit 20GB --nthreads 1 --nprocs <PROCS>"


TOTAL_NODES = 10



# from dask_mpi import initialize
# initialize(interface="enp175s0f0", local_directory="/scratch/dnperera/dask/", nthreads=1)


# cluster = SSHCluster(
#      ["v-001", "v-002", "v-003", "v-004"],
#      connect_options={"known_hosts": "/N/u2/d/dnperera/.ssh/known_hosts", "username": "dnperera",},
#      worker_options={"nthreads": 2},
#      scheduler_options={"port": 8786, "dashboard_address": ":8797"},
#     remote_python = "/N/u2/d/dnperera/victor/git/twisterx/ENV/bin/python"
#  )



# client = Client("172.29.200.201:8786")  # Connect this local process to remote work
# client = Client(cluster)
# client = Client("172.29.200.201:8786", scheduler_file="/N/u2/d/dnperera/victor/git/twisterx/cpp/src/experiments/sched.json")


# ssh v-002 ~/victor/git/twisterx/ENV/bin/dask-scheduler --interface enp175s0f0 --scheduler-file ~/dask-sched.json 
# ssh IP ~/victor/git/twisterx/ENV/bin/dask-worker v-002:8786 --interface enp175s0f0 --nthreads 1 --nprocs PROCS --memory-limit 10GB --local-directory /scratch/dnperera/dask/ --scheduler-file ~/dask-sched.json 

#ssh v-002 "pkill dask-scheduler "
# ssh v-002 "pkill dask-worker" 

nodes_file = "nodes"
ips = []
with open(nodes_file, 'r') as fp:
     for l in fp.readlines():
        ips.append(l.split(' ')[0])
    
    
def start_dask(procs, nodes):
    print("starting scheduler", flush=True)
    subprocess.Popen(["ssh", "v-001", "/N/u2/d/dnperera/victor/git/twisterx/ENV/bin/dask-scheduler", "--interface", "enp175s0f0", "--scheduler-file", "/N/u2/d/dnperera/dask-sched.json"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    time.sleep(5)   

    
    for ip in ips[0:nodes]:
        print("starting worker", ip, flush=True)
        subprocess.Popen(["ssh", ip, "/N/u2/d/dnperera/victor/git/twisterx/ENV/bin/dask-worker", "v-001:8786", "--interface", "enp175s0f0", "--nthreads", "1", "--nprocs", str(procs), "--memory-limit", "20GB", "--local-directory", "/scratch/dnperera/dask/", "--scheduler-file", "/N/u2/d/dnperera/dask-sched.json"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    time.sleep(5)   
        
        

def stop_dask():   
    for ip in ips:
        print("stopping worker", ip, flush=True)
        subprocess.run(["ssh", ip, "pkill",  "dask-worker"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)      
    
    time.sleep(5)           
    
    print("stopping scheduler", flush=True)
    subprocess.run(["pkill", "dask-scheduler"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(5)   
    

for r in rows:
    for w in world:
        procs = int(math.ceil(w / TOTAL_NODES))
        print("procs per worker", procs, flush=True)

        assert procs <= 16
        
        stop_dask()
        start_dask(procs, min(w, TOTAL_NODES))
        
        client = Client("v-001:8786") 
        
        df_l = dd.read_csv(f"~/temp/twx/{scale}/{r}/{w}/csv1_*.csv")
        df_r = dd.read_csv(f"~/temp/twx/{scale}/{r}/{w}/csv2_*.csv")

        client.persist([df_l, df_r])

        print("left rows", len(df_l), flush=True)
        print("right rows", len(df_r), flush=True)

        try:
            for i in range(it):
                t1 = time.time()
                out = df_l.merge(df_r, on='0', how='inner', suffixes=('_left','_right')).compute()
                t2 = time.time()

                print(f"###time {r} {w} {i} {(t2 - t1)*1000:.0f}, {len(out)}", flush=True)
        
            client.restart()

            client.close()
        finally:
            stop_dask()