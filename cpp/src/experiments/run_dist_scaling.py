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
parser.add_argument('--no-pspark', dest='no_pyspark', action='store_true', help='skip pyspark')
parser.add_argument('--no-twx', dest='no_twx', action='store_true', help='skip twx')
parser.add_argument('--no-ptwx', dest='no_ptwx', action='store_true', help='skip twx')
parser.add_argument('--no-jtwx', dest='no_jtwx', action='store_true', help='skip twx')

parser.add_argument('-s', required=True, dest='scaling', type=str, help='weak or strong')
parser.add_argument('-r', dest='rows', type=float, nargs='+', help='row cases in millions',
                    required=True)
parser.add_argument('-w', dest='world', type=int, nargs='+', help='world sizes',
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 160])
parser.add_argument('--reps', dest='reps', type=int, help='number of repetitions', default=4)
parser.add_argument('-t', dest='threads', type=int, help='file writing threads', default=16)

args = parser.parse_args()
args = vars(args)

dry = args['dry']
execs = args['execs']
spark = not args['no_spark']
pspark = not args['no_pyspark']
twx = not args['no_twx']
jtwx = not args['no_jtwx']
ptwx = not args['no_ptwx']

row_cases = [int(ii * 1000000) for ii in args['rows']]
repetitions = args['reps']
world_sizes = args['world']

if args['scaling'] == 'w':
    scaling = 'weak'
elif args['scaling'] == 's':
    scaling = 'strong'
else:
    scaling = None
    print("\n\nnothing to do!", flush=True)
    exit(0)

if dry:
    row_cases = [20]
    world_sizes = [1, 2, 4]
    repetitions = 1

print("##### args: ", args, flush=True)

# if not twx and not spark:
#     print("\n\nnothing to do!", flush=True)
#     exit(0)

print(f"##### running {execs} test for {scaling} scaling", flush=True)

cols = 4
key_duplication_ratio = 0.99  # on avg there will be rows/key_range_ratio num of duplicate keys

print("##### repetitions for each test", repetitions, flush=True)

hdfs_url = "hdfs://v-login1:9001"
hdfs_dfs = f"~/victor/software/hadoop-2.10.0//bin/hdfs dfs -fs {hdfs_url}"
dfs_base = "/twx/"

spark_home = "~/victor/software/spark-2.4.6-bin-hadoop2.7"
spark_submit = f"{spark_home}/bin/spark-submit "
spark_jar = "~/victor/git/SparkOps/target/scala-2.11/sparkops_2.11-0.1.jar "
spark_py = "~/victor/git/cylon/cpp/src/experiments/pyspark_run.py "
spark_master = "spark://v-001:7077"
spark_slaves_file = "/N/u2/d/dnperera/victor/software/spark-2.4.6-bin-hadoop2.7/conf/slaves"

def cleanup_hdfs():
    print("\n\n##### cleaning up hdfs dfs", flush=True)
    os.system(f"{hdfs_dfs} -rm -skipTrash {dfs_base}/csv*.csv")
    
# cleanup_hdfs()

PYTHON_EXEC = "~/victor/git/cylon/ENV/bin/python"

TOTAL_NODES = 10
TOTAL_MEM = 200 # GB

def spark_write_slaves_file(n):
    with open(spark_slaves_file, 'w') as f:
        for i in range(n):
            f.write(f"v-{i+1:03}")
            f.write("\n")
    
    os.system(f"cat {spark_slaves_file}")
        


def restart_spark_cluster(world_size):
    print(f"\n\n##### restarting spark cluster. world size {world_size}!", flush=True)

    spark_write_slaves_file(TOTAL_NODES)
    os.system(f"{spark_home}/sbin/stop-all.sh")
    print(f"##### spark cluster stopped!", flush=True)

    cores_per_worker = int(math.ceil(world_size / TOTAL_NODES))
    print(f"\n\n##### new cores per worker {cores_per_worker}!", flush=True)
    os.system(
        f"sed -i 's/SPARK_WORKER_CORES=[[:digit:]]\\+/SPARK_WORKER_CORES={cores_per_worker}/g' "
        f" ~/.bashrc ")
    
    if world_size < 10:
        spark_write_slaves_file(world_size)
    else:
        spark_write_slaves_file(TOTAL_NODES)

    sleep(5)
    os.system(f"tail ~/.bashrc")

    os.system(f"{spark_home}/sbin/start-all.sh")
    print(f"##### spark cluster restarted! {world_size}", flush=True)

    
THREADS = 20


def push_file_to_hdfs(f):
    os.system(f"{hdfs_dfs} -put -f {f} {dfs_base}")


from multiprocessing import Pool


for i in row_cases:
    for w in world_sizes:
        print(f"\n##### rows {i} world_size {w} starting!", flush=True)
        s_dir = f"/N/u2/d/dnperera/temp/twx/{scaling}/{i}/{w}/"
        b_dir = f"/scratch_hdd/dnperera/"

#         if spark or pspark:
#             print(f"pushing files to hdfs {i}", flush=True)
#             all_files = [os.path.abspath(os.path.join(s_dir, p)) for p in os.listdir(s_dir)]
#             print(f"total files {len(all_files)}", flush=True)

#             p = Pool(THREADS)
#             p.map(push_file_to_hdfs, all_files)
#             p.close()
#             p.join()
#             print(f"pushing files to hdfs DONE", flush=True)

#             print(f"pushing files to hdfs {i}")
#             os.system(f"{hdfs_dfs} -put -f {s_dir}/csv*.csv {dfs_base}")

        if twx:
            for ex in execs:
                if dry:
                    join_exec = f"mpirun -np {w} ../../../build/bin/{ex} dry"
                else:
                    hostfile = "" if w == 1 else "--hostfile nodes"
                    join_exec = f"mpirun --map-by node --report-bindings -mca btl vader,tcp,openib," \
                                f"self -mca btl_tcp_if_include enp175s0f0 --mca btl_openib_allow_ib 1 " \
                                f"{hostfile} --bind-to core --bind-to socket -np {w} " \
                                f"../../../build/bin/{ex} {s_dir} {b_dir}"
                print("\n\n##### running", join_exec, flush=True)

                for r in range(repetitions):
                    print(f"\n\n{ex} {i} {w} ##### twx {r + 1}/{repetitions} iter start! "
                          f"SPLIT_FROM_HERE", flush=True)
                    os.system(f"{join_exec}")
        
        if ptwx:
            for ex in execs:
                if ex != "table_join_dist_test":
                    print("not suppoted", ex, flush=True)
                    continue
                
                hostfile = "" if w == 1 else "--hostfile nodes"
                join_exec = f"mpirun --map-by node --report-bindings -mca btl vader,tcp,openib," \
                            f"self -mca btl_tcp_if_include enp175s0f0 --mca btl_openib_allow_ib 1 " \
                            f"{hostfile} --bind-to core --bind-to socket -np {w} " \
                            f"{PYTHON_EXEC} ../../../python/examples/experiments/table_join_dist_test.py -s {s_dir} -b {b_dir}"
                print("\n\n##### running", join_exec, flush=True)

                for r in range(repetitions):
                    print(f"\n\n{ex} {i} {w} ##### ptwx {r + 1}/{repetitions} iter start! "
                          f"SPLIT_FROM_HERE", flush=True)
                    os.system(f"{join_exec}")
                    
        if jtwx:
            for ex in execs:
                if ex != "table_join_dist_test":
                    print("not suppoted", ex, flush=True)
                    continue
                
                hostfile = "" if w == 1 else "--hostfile nodes"
                join_exec = f"mpirun --map-by node --report-bindings -mca btl vader,tcp,openib," \
                            f"self -mca btl_tcp_if_include enp175s0f0 --mca btl_openib_allow_ib 1 " \
                            f"{hostfile} --bind-to socket -np {w} " \
                            f"java -Xmx4g -Xms4g -cp ~/victor/git/cylon/java/target/cylon-0.1-SNAPSHOT-jar-with-dependencies.jar org.cylon.examples.DistributedJoinExample {s_dir} {b_dir}"
                print("\n\n##### running", join_exec, flush=True)

                for r in range(repetitions):
                    print(f"\n\n{ex} {i} {w} ##### jtwx {r + 1}/{repetitions} iter start! "
                          f"SPLIT_FROM_HERE", flush=True)
                    os.system(f"{join_exec}")


        if spark:
            restart_spark_cluster(w)
            for ex in execs:
                print(f"\n\n##### starting spark rows {i} world_size {w}...", flush=True)
                spark_exec = f"{spark_submit} --class {ex} {spark_jar} {w} {hdfs_url}/{dfs_base} " \
                             f"{spark_master}"
                print("\n##### executing", spark_exec, flush=True)

                for r in range(repetitions):
                    print(f"\n\n{ex} {i} {w} ##### spark {r + 1}/{repetitions} iter start! "
                          f"SPLIT_FROM_HERE", flush=True)
                    os.system(spark_exec)

            print("\n\n##### cleaning up hdfs dfs", flush=True)
            os.system(f"{hdfs_dfs} -rm -skipTrash {dfs_base}/csv*.csv")
            print("\n\n##### spark done .....", flush=True)
            
        if pspark:
            restart_spark_cluster(w)
            cores_per_worker = int(math.ceil(w / TOTAL_NODES))
            mem_per_core = int(TOTAL_MEM/cores_per_worker)
            for ex in execs:
                print(f"\n\n##### starting pyspark rows {i} world_size {w}...", flush=True)
                spark_exec = f"{spark_submit} {spark_py} {w} {hdfs_url}/{dfs_base} " \
                             f"{spark_master} {repetitions} {mem_per_core}"
                print("\n##### executing", spark_exec, flush=True)

                os.system(spark_exec)
            
#             cleanup_hdfs()

            print("\n\n##### spark done .....", flush=True)
        
        print(f"\n##### rows {i} world_size {w} done!\n-----------------------------------------", flush=True)
        
    print(f"\n\n##### rows {i} done!\n ====================================== \n", flush=True)
