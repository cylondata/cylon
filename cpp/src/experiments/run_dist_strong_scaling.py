import os
from os.path import expanduser

from generate_csv import generate_file
from multiprocessing import Process

import argparse

parser = argparse.ArgumentParser(description='generate random data')
parser.add_argument('-e', required=True, dest='exec', type=str, help='executable')
parser.add_argument('--dry', action='store_true', help='if this is a dry run')

args = parser.parse_args()
args = vars(args)

dry = args['dry']
exec = args['exec']

home = expanduser("~")

base_dir = "~/temp"

csvs = [f"{base_dir}/csv1_RANK.csv", f"{base_dir}/csv2_RANK.csv"]
if dry:
    row_cases = [20]
    world_sizes = [1, 2, 4]
    repetitions = 1
else:
    row_cases = [int(ii * 1000000) for ii in [50, 100, 200, 400, 500]]
    world_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 160]
    repetitions = 4

print(f"\n##### running {exec} test for strong scaling", flush=True)

out_dir = f"{base_dir}/{exec}/"
print(f"\n##### output dir: {out_dir}", flush=True)
os.system(f"rm -rf {out_dir}; mkdir -p {out_dir}")

cols = 4
key_duplication_ratio = 0.99  # on avg there will be rows/key_range_ratio num of duplicate keys

print("\n##### repetitions for each test", repetitions, flush=True)

file_gen_threads = 16


def generate_files(_rank, _i, _krange):
    print(f"generating files for {_i} {_rank}")
    for _f in csvs:
        generate_file(output=_f.replace('RANK', str(_rank)), rows=_i, cols=cols, krange=_krange)


# for i in [10000000]:
for i in row_cases:
    # test_dir = f"{out_dir}/{i}"
    # os.system(f"rm -rf {test_dir}; mkdir -p {test_dir}")

    for w in world_sizes:
        krange = (0, int(i * key_duplication_ratio))

        # generate 2 cvs for world size
        print(f"\n\n##### generating files of rows {i} {w}!", flush=True)
        for rank in range(0, w, file_gen_threads):
            procs = []
            for r in range(rank, min(rank + file_gen_threads, w)):
                p = Process(target=generate_files, args=(r, int(i/w), krange))
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

        print(f"\n\n##### rows {i} world_size {w} starting!", flush=True)

        if dry:
            join_exec = f"mpirun -np {w} ../../../build/bin/{exec} dry"
        else:
            hostfile = "" if w == 1 else "--hostfile nodes"
            join_exec = f"mpirun --map-by node --report-bindings -mca btl vader,tcp,openib," \
                        f"self -mca btl_tcp_if_include enp175s0f0 --mca btl_openib_allow_ib 1 " \
                        f"{hostfile} -np {w} ../../../build/bin/{exec}"
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
