import argparse
import gc
import math
import os
import time

import pandas as pd

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
nodes_file = "nodes"
ips = []

INIT = False


# awk '(NR == 1) || (FNR > 1)' csv2_*.csv > csv2_merged.csv
def merge_files(prefix):
    merged_f = prefix + "merged.csv"
    if not os.path.exists(merged_f):
        print(merged_f + " does not exist!", flush=True)
        s = "awk '(NR == 1) || (FNR > 1)' " + prefix + "*.csv > " + merged_f
        print(s, flush=True)
        os.system(s)

    return merged_f


if __name__ == "__main__":
    for r in rows:
        for w in world:
            procs = int(math.ceil(w / TOTAL_NODES))
            print("procs per worker", procs, " iter ", it, flush=True)

            assert procs <= 16

            if w > 1:
                raise Exception("sss")
            else:
                f_l = f"/N/u2/d/dnperera/temp/twx/{scale}/{r}/{w}/csv1_0.csv"
                f_r = f"/N/u2/d/dnperera/temp/twx/{scale}/{r}/{w}/csv2_0.csv"

            df_l = pd.read_csv(f_l)
            df_r = pd.read_csv(f_r)

            for i in range(it):
                t1 = time.time()
                out = df_l.merge(df_r, on='0', how='inner', suffixes=('_left', '_right'))
                t2 = time.time()

                print(f"###time {r} {w} {i} {(t2 - t1) * 1000:.0f}, {out.shape[0]}", flush=True)

                del out
                gc.collect()
