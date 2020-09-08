import os


import time
import argparse
import math
import subprocess
import os 
import gc


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

TOTAL_NODES = 10
RAY_PW = '1234'
RAY_EXEC = "/N/u2/d/dnperera/victor/git/cylon/ENV/bin/ray"

nodes_file = "nodes"
ips = []

INIT = False

with open(nodes_file, 'r') as fp:
    for l in fp.readlines():
        ips.append(l.split(' ')[0])


# ray start --head --redis-port=6379 --node-ip-address=v-001
def start_ray(procs, nodes):
    print("starting head", flush=True)
    query = ["ssh", "v-001", RAY_EXEC, "start",
             "--head", "--redis-port=6379", "--node-ip-address=v-001",
             f"--redis-password={RAY_PW}", f"--num-cpus={procs}",
#              f"--memory={20 * procs * (10 ** 9)}"]
#              "--resources={\"memory\":" + str(20 * procs * 10 ** 9) +"}"
            ]
    print("running: ", " ".join(query), flush=True)
    subprocess.Popen(query, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    time.sleep(5)

    for ip in ips[1:nodes]:
        print("starting worker", ip, flush=True)
        query = ["ssh", ip, RAY_EXEC, "start",
                 "--redis-address=\'v-001:6379\'", f"--node-ip-address={ip}",
                 f"--redis-password={RAY_PW}", f"--num-cpus={procs}",
#                  f"--memory={20 * procs * 10 ** 9}"]
#                  "--resources={\"memory\":" + str(20 * procs * 10 ** 9) +"}"
                ]
        print("running: ", " ".join(query), flush=True)
        subprocess.Popen(query, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    time.sleep(5)

    import ray
    ray.init(redis_address='v-001:6379', redis_password=RAY_PW)


def stop_ray():
    import ray
    ray.shutdown()

    print("stopping workers", flush=True)
    for ip in ips:
        subprocess.run(["ssh",ip , "pkill", "-f", "ray"], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    time.sleep(5)
#     global INIT
#     INIT = False


def initialize():
    global INIT
    if not INIT:
        import ray
        ray.init(redis_address='v-001:6379', redis_password=RAY_PW)
        INIT = True


#awk '(NR == 1) || (FNR > 1)' csv2_*.csv > csv2_merged.csv
def merge_files(prefix):
    merged_f = prefix + "merged.csv"
    if not os.path.exists(merged_f):
        print(merged_f +" does not exist!", flush=True)        
        s = "awk '(NR == 1) || (FNR > 1)' " + prefix + "*.csv > " + merged_f
        print(s, flush=True)
        os.system(s)
    
    return merged_f


if __name__ == "__main__":
    os.environ["MODIN_ENGINE"] = "ray"  # Modin will use Ray

    for r in rows:
        for w in world:
            procs = int(math.ceil(w / TOTAL_NODES))
            print("procs per worker", procs, " iter ", it, flush=True)

            assert procs <= 16

            try:
                stop_ray()
                start_ray(procs, min(w, TOTAL_NODES))
                
#                 initialize()

                import modin.pandas as pd
                pd.DEFAULT_NPARTITIONS = w

                if w > 1:
                    f_l = merge_files(f"/N/u2/d/dnperera/temp/twx/{scale}/{r}/{w}/csv1_")
                    f_r = merge_files(f"/N/u2/d/dnperera/temp/twx/{scale}/{r}/{w}/csv2_")
                else:
                    f_l = f"/N/u2/d/dnperera/temp/twx/{scale}/{r}/{w}/csv1_0.csv"
                    f_r = f"/N/u2/d/dnperera/temp/twx/{scale}/{r}/{w}/csv2_0.csv"
                
                for i in range(it):

                    df_l = pd.read_csv(f_l)
                    df_r = pd.read_csv(f_r)


                    t1 = time.time()
                    out = df_l.merge(df_r, on='0', how='inner', suffixes=('_left', '_right'))
                    t2 = time.time()

                    print(f"###time {r} {w} {i} {(t2 - t1) * 1000:.0f}, {out.shape[0]}", flush=True)
                    
                    del df_l 
                    del df_r
                    del out 
                    gc.collect()
            finally:
                stop_ray()
