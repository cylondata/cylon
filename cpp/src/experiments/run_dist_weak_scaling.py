import os
from os.path import expanduser

from multiprocessing import Process
import math
import argparse
from time import sleep

parser = argparse.ArgumentParser(description='generate random data')
parser.add_argument('-e', required=True, dest='execs', type=str, nargs='+', help='executables')
parser.add_argument('--dry', action='store_true', help='if this is a dry run')
parser.add_argument('--no-spark', dest='no_spark', action='store_true', help='skip spark')
parser.add_argument('--no-twx', dest='no_twx', action='store_true', help='skip twx')
parser.add_argument('-r', dest='rows', type=float, nargs='+', help='row cases in millions',
                    default=[0.5, 1, 2])
parser.add_argument('-w', dest='world', type=int, nargs='+', help='world sizes',
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 160])
parser.add_argument('--reps', dest='reps', type=int, help='number of repetitions', default=4)
parser.add_argument('-t', dest='threads', type=int, help='file writing threads', default=16)

args = parser.parse_args()
args = vars(args)

dry = args['dry']
execs = args['execs']
spark = not args['no_spark']
twx = not args['no_twx']
row_cases = [int(ii * 1000000) for ii in args['rows']]
repetitions = args['reps']
world_sizes = args['world']

if dry:
    row_cases = [20]
    world_sizes = [1, 2, 4]
    repetitions = 1

print("\n\n##### args: ", args, flush=True)

if not twx and not spark:
    print("\n\nnothing to do!", flush=True)
    exit(0)

home = expanduser("~")

base_dir = "~/temp"
print("\n\n##### cleaning up .....", flush=True)
os.system(f"mkdir -p {base_dir}; rm -f {base_dir}/*.csv")

csvs = [f"{base_dir}/csv1_RANK.csv", f"{base_dir}/csv2_RANK.csv"]

print(f"\n##### running {execs} test for weak scaling", flush=True)

# out_dir = f"{base_dir}/{ex}/"
# print(f"\n##### output dir: {out_dir}", flush=True)
# os.system(f"rm -rf {out_dir}; mkdir -p {out_dir}")

cols = 4
key_duplication_ratio = 0.99  # on avg there will be rows/key_range_ratio num of duplicate keys

print("\n##### repetitions for each test", repetitions, flush=True)

file_gen_threads = args['threads']

hdfs_url = "hdfs://v-login1:9001"
hdfs_dfs = f"~/victor/software/hadoop-2.10.0//bin/hdfs dfs -fs {hdfs_url}"
dfs_base = "/twx/"

spark_home = "~/victor/software/spark-2.4.6-bin-hadoop2.7"
spark_submit = f"{spark_home}/bin/spark-submit "
spark_jar = "~/victor/git/SparkOps/target/scala-2.11/sparkops_2.11-0.1.jar "
spark_master = "spark://v-001:7077"

print("\n\n##### cleaning up hdfs dfs", flush=True)
os.system(f"{hdfs_dfs} -rm -skipTrash {dfs_base}/csv*.csv")

TOTAL_NODES = 10


def generate_files(_rank, _i, _krange):
    # print(f"generating files for {_i} {_rank}")
    for _f in csvs:
        os.system(f"python generate_csv.py -o {_f.replace('RANK', str(_rank))} -r {_i} -c {cols} "
                  f"--krange 0 {_krange[1]}")

    if spark:  # push all csvs to hdfs
        print(f"pushing files to hdfs {_i} {_rank}")
        os.system(f"{hdfs_dfs} -put -f {base_dir}/csv*_{_rank}.csv {dfs_base}")


def restart_spark_cluster(world_size):
    print(f"\n\n##### restarting spark cluster. world size {world_size}!", flush=True)

    os.system(f"{spark_home}/sbin/stop-all.sh")
    print(f"##### spark cluster stopped!", flush=True)

    cores_per_worker = int(math.ceil(world_size / TOTAL_NODES))
    print(f"\n\n##### new cores per worker {cores_per_worker}!", flush=True)
    os.system(
        f"sed -i 's/SPARK_WORKER_CORES=[[:digit:]]\\+/SPARK_WORKER_CORES={cores_per_worker}/g' "
        f" ~/.bashrc ")

    sleep(5)
    os.system(f"tail ~/.bashrc")

    os.system(f"{spark_home}/sbin/start-all.sh")
    print(f"##### spark cluster restarted! {world_size}", flush=True)


# for i in [10000000]:
for i in row_cases:
    # test_dir = f"{out_dir}/{i}"
    # os.system(f"rm -rf {test_dir}; mkdir -p {test_dir}")

    for w in world_sizes:
        krange = (0, int(i * key_duplication_ratio * w))

        # generate 2 cvs for world size
        print(f"\n\n##### generating files of rows {i} {w}!", flush=True)
        for rank in range(0, w, file_gen_threads):
            procs = []
            for r in range(rank, min(rank + file_gen_threads, w)):
                p = Process(target=generate_files, args=(r, i, krange))
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

        print(f"\n\n##### rows {i} world_size {w} starting!", flush=True)

        if spark:
            restart_spark_cluster(w)

        if twx:
            for ex in execs:
                if dry:
                    join_exec = f"mpirun -np {w} ../../../build/bin/{ex} dry"
                else:
                    hostfile = "" if w == 1 else "--hostfile nodes"
                    join_exec = f"mpirun --map-by node --report-bindings -mca btl vader,tcp,openib," \
                                f"self -mca btl_tcp_if_include enp175s0f0 --mca btl_openib_allow_ib 1 " \
                                f"{hostfile} --bind-to core --bind-to socket -np {w} " \
                                f"../../../build/bin/{ex}"
                print("\n\n##### running", join_exec, flush=True)

                for r in range(repetitions):
                    print(f"\n\n{ex} {i} {w} ##### twx {r + 1}/{repetitions} iter start! "
                          f"SPLIT_FROM_HERE", flush=True)
                    os.system(f"{join_exec}")

            print(
                f"\n\n##### rows {i} world_size {w} done!\n-----------------------------------------",
                flush=True)

        if spark:
            for ex in execs:
                print(f"\n\n##### starting spark rows {i} world_size {w}...", flush=True)
                spark_exec = f"{spark_submit} --class {ex} {spark_jar} {w} {hdfs_url}/{dfs_base} " \
                             f"{spark_master}"
                print("\n\n##### executing", spark_exec, flush=True)

                for r in range(repetitions):
                    print(f"\n\n{ex} {i} {w} ##### spark {r + 1}/{repetitions} iter start! "
                          f"SPLIT_FROM_HERE", flush=True)
                    os.system(spark_exec)

            print("\n\n##### cleaning up hdfs dfs", flush=True)
            os.system(f"{hdfs_dfs} -rm -skipTrash {dfs_base}/csv*.csv")
            print("\n\n##### spark done .....", flush=True)

        print("\n\n##### cleaning up .....", flush=True)
        os.system(f"rm -f {base_dir}/*.csv")

    print(f"\n\n##### rows {i} done!\n ====================================== \n", flush=True)
