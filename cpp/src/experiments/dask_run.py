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

TOTAL_MEM = 200
MIN_MEM= 20 
TOTAL_NODES = 10

nodes_file = "nodes"
ips = []
with open(nodes_file, 'r') as fp:
     for l in fp.readlines():
        ips.append(l.split(' ')[0])
    
    
def start_dask(procs, nodes):
    print("starting scheduler", flush=True)
    subprocess.Popen(["ssh", "v-001", "/N/u2/d/dnperera/victor/git/cylon/ENV/bin/dask-scheduler", "--interface", "enp175s0f0", "--scheduler-file", "/N/u2/d/dnperera/dask-sched.json"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    time.sleep(5)   

    mem_per_worker = max(int(TOTAL_MEM/procs), MIN_MEM)
    
    for ip in ips[0:nodes]:
        print("starting worker", ip, flush=True)
        subprocess.Popen(["ssh", ip, "/N/u2/d/dnperera/victor/git/cylon/ENV/bin/dask-worker",
                          "v-001:8786", "--interface", "enp175s0f0", "--nthreads", "1",
                          "--nprocs", str(procs), "--memory-limit", f"{mem_per_worker}GB",
                          "--local-directory", "/scratch/dnperera/dask/", "--scheduler-file", "/N/u2/d/dnperera/dask-sched.json"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    time.sleep(5)   
        
        
def stop_dask():   
    for ip in ips:
        print("stopping worker", ip, flush=True)
        subprocess.run(["ssh", ip, "pkill", "-f", "dask-worker"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)      
    
    time.sleep(5)           
    
    print("stopping scheduler", flush=True)
    subprocess.run(["pkill", "-f", "dask-scheduler"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(5)   
    

for r in rows:
    for w in world:
        procs = int(math.ceil(w / TOTAL_NODES))
        print("procs per worker", procs, flush=True)

        assert procs <= 16
        
        stop_dask()
        start_dask(procs, min(w, TOTAL_NODES))
        
        client = Client("v-001:8786") 
        
        df_l = dd.read_csv(f"~/temp/twx/{scale}/{r}/{w}/csv1_*.csv").repartition(npartitions=w)
        df_r = dd.read_csv(f"~/temp/twx/{scale}/{r}/{w}/csv2_*.csv").repartition(npartitions=w)

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