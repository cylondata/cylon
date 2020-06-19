import os
from os.path import expanduser

from generate_csv import generate_file

home = expanduser("~")
# join_exec = f"{home}/git/twisterx/build/bin/table_api_test_hash"
# print(f"twx home: {join_exec}", flush=True)

base_dir = "~/temp"

csvs = [f"{base_dir}/csv1_RANK.csv", f"{base_dir}/csv2_RANK.csv"]
row_cases = [int(ii * 1000000) for ii in [0.1, 0.25, 0.5, 0.75, 1]]
#row_cases = [10, 20]

world_sizes = [2, 4, 8, 16, 32, 64]
#world_sizes = [2, 4, ]

out_dir = f"{base_dir}/twx_join_test/"
print(f"\n##### output dir: {out_dir}", flush=True)
os.system(f"rm -rf {out_dir}; mkdir -p {out_dir}")

cols = 4
key_duplication_ratio = 0.99  # on avg there will be rows/key_range_ratio num of duplicate keys

repetitions = 3
print("\n##### repetitions for each test", repetitions, flush=True)

# for i in [10000000]:
for i in row_cases:
    # test_dir = f"{out_dir}/{i}"
    # os.system(f"rm -rf {test_dir}; mkdir -p {test_dir}")

    krange = (0, int(i * key_duplication_ratio))

    # generate 2 cvs for max of world
    for rank in range(max(world_sizes)):
        for f in csvs:
            generate_file(output=f.replace('RANK', str(rank)), rows=i, cols=cols, krange=krange)

    for w in world_sizes:
        print(f"\n\n##### rows {i} world_size {w} starting!", flush=True)

        join_exec = f"mpirun --map-by node --report-bindings -mca btl vader,tcp,openib," \
                    f"self -mca btl_tcp_if_include enp175s0f0 --mca btl_openib_allow_ib 1 " \
                    f"--hostfile nodes -np {w} ../../../build/bin/table_join_dist_test"
        # join_exec = f"mpirun -np {w} ../../../build/bin/table_join_dist_test"
        print("\n\n##### running", join_exec, flush=True)

        for r in range(repetitions):
            os.system(f"{join_exec}")
            print(f"\n\n##### {r + 1}/{repetitions} iter done!", flush=True)

        # os.system(f"mv {csv1} {test_dir}")
        # os.system(f"mv {csv2} {test_dir}")
        # for j in ['right', 'left', 'inner', 'outer']:
        #     os.system(f"mv /tmp/h_out_{j}.csv {test_dir}/")
        #     os.system(f"mv /tmp/s_out_{j}.csv {test_dir}/")

        print(f"\n\n##### rows {i} world_size {w} done!\n-----------------------------------------",
              flush=True)
