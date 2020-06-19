import os
from os.path import expanduser

from generate_csv import generate_file
import argparse

parser = argparse.ArgumentParser(description='generate random data')
parser.add_argument('--dry', action='store_true', help='if this is a dry run')

args = parser.parse_args()
args = vars(args)

dry = args['dry']

home = expanduser("~")
# join_exec = f"{home}/git/twisterx/build/bin/table_api_test_hash"
# print(f"twx home: {join_exec}", flush=True)

base_dir = "~/temp"

csvs = [f"{base_dir}/csv1_RANK.csv", f"{base_dir}/csv2_RANK.csv"]
if dry:
    row_cases = [10]
    world_sizes = [1, 2, ]
    repetitions = 1
else:
    row_cases = [int(ii * 1000000) for ii in [0.125, 0.25, 0.5, 1, 2]]
    world_sizes = [1, 2, 4, 8, 16, 32, 64]
    repetitions = 4

out_dir = f"{base_dir}/twx_join_test/"
print(f"\n##### output dir: {out_dir}", flush=True)
os.system(f"rm -rf {out_dir}; mkdir -p {out_dir}")

cols = 4
key_duplication_ratio = 0.99  # on avg there will be rows/key_range_ratio num of duplicate keys

print("\n##### repetitions for each test", repetitions, flush=True)

# for i in [10000000]:
for i in row_cases:
    # test_dir = f"{out_dir}/{i}"
    # os.system(f"rm -rf {test_dir}; mkdir -p {test_dir}")

    for w in world_sizes:
        krange = (0, int(i * key_duplication_ratio * w))

        # generate 2 cvs for world size
        print(f"\n\n##### generating files of rows {i}!", flush=True)
        for rank in range(w):
            for f in csvs:
                generate_file(output=f.replace('RANK', str(rank)), rows=i, cols=cols, krange=krange)

        print(f"\n\n##### rows {i} world_size {w} starting!", flush=True)

        if dry:
            join_exec = f"mpirun -np {w} ../../../build/bin/table_join_dist_test"
        else:
            join_exec = f"mpirun --map-by node --report-bindings -mca btl vader,tcp,openib," \
                        f"self -mca btl_tcp_if_include enp175s0f0 --mca btl_openib_allow_ib 1 " \
                        f"--hostfile nodes -np {w} ../../../build/bin/table_join_dist_test"
        print("\n\n##### running", join_exec, flush=True)

        for r in range(repetitions):
            print(f"\n\n{i} {w} ##### {r + 1}/{repetitions} iter start!", flush=True)
            os.system(f"{join_exec}")

        # os.system(f"mv {csv1} {test_dir}")
        # os.system(f"mv {csv2} {test_dir}")
        # for j in ['right', 'left', 'inner', 'outer']:
        #     os.system(f"mv /tmp/h_out_{j}.csv {test_dir}/")
        #     os.system(f"mv /tmp/s_out_{j}.csv {test_dir}/")

        print(f"\n\n##### rows {i} world_size {w} done!\n-----------------------------------------",
              flush=True)

    print(f"\n\n##### rows {i} done!\n ====================================== \n", flush=True)
