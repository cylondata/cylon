import os


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
             f"--memory={20 * procs * (10 ** 9)}"]
    print("running: ", " ".join(query), flush=True)
    subprocess.Popen(query, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    time.sleep(5)

    for ip in ips[1:nodes]:
        print("starting worker", ip, flush=True)
        query = ["ssh", ip, RAY_EXEC, "start",
                 "--address=\'v-001:6379\'", f"--node-ip-address={ip}",
                 f"--redis-password={RAY_PW}", f"--num-cpus={procs}",
                 f"--memory={20 * procs * 10 ** 9}"]
        print("running: ", " ".join(query), flush=True)
        subprocess.Popen(query, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    time.sleep(5)

    import ray
    ray.init(address='v-001:6379', redis_password=RAY_PW)


def stop_ray():
    for ip in ips:
        print("stopping worker", ip, flush=True)
        subprocess.run(["ssh",ip , "pkill", "-f", "ray"], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)

    time.sleep(5)
    global INIT
    INIT = False


def initialize():
    global INIT
    if not INIT:
        import ray
        ray.init(address='v-001:6379', redis_password=RAY_PW)
        INIT = True


if __name__ == "__main__":
    for r in rows:
        for w in world:
            procs = int(math.ceil(w / TOTAL_NODES))
            print("procs per worker", procs, flush=True)

            assert procs <= 16

            stop_ray()
            start_ray(procs, min(w, TOTAL_NODES))

            # initialize()

            import modin.pandas as pd

            df_l = pd.read_csv(f"~/temp/twx/{scale}/{r}/{w}/csv1_*.csv").repartition(npartitions=w)
            df_r = pd.read_csv(f"~/temp/twx/{scale}/{r}/{w}/csv2_*.csv").repartition(npartitions=w)
            #
            # client.persist([df_l, df_r])
            #
            # print("left rows", len(df_l), flush=True)
            # print("right rows", len(df_r), flush=True)

            try:
                for i in range(it):
                    t1 = time.time()
                    out = df_l.merge(df_r, on='0', how='inner', suffixes=('_left', '_right'))
                    t2 = time.time()

                    print(f"###time {r} {w} {i} {(t2 - t1) * 1000:.0f}, {len(out)}", flush=True)
            finally:
                stop_ray()
