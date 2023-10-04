import time
import argparse

import pandas as pd
from mpi4py import MPI
from numpy.random import default_rng
from pycylon.frame import CylonEnv, DataFrame
from pycylon.net import MPIConfig
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.common.dotdict import dotdict
from cloudmesh.common.Shell import Shell
from pycylon.net.ucc_config import UCCConfig
from pycylon.net.redis_ucc_oob_context import UCCRedisOOBContext
from pycylon.net.reduce_op import ReduceOp


def cylon_join(data=None):
    global ucc_config
    StopWatch.start(f"join_total_{data['host']}_{data['rows']}_{data['it']}")

    redis_context = UCCRedisOOBContext(data['world_size'], f"tcp://{data['redis_host']}:{data['redis_port']}")

    if redis_context is not None:
        ucc_config = UCCConfig(redis_context)

    if ucc_config is None:
        print("unable to initialize uccconfig")

    env = CylonEnv(config=ucc_config, distributed=True)

    u = data['unique']

    context = env.context

    if context is None:
        print("unable to retrieve cylon context")

    communicator = context.get_communicator()

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * env.world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / env.world_size)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
    data2 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))

    if env.rank == 0:
        print("Task# ", data['task'])

    for i in range(data['it']):
        env.barrier()
        StopWatch.start(f"join_{i}_{data['host']}_{data['rows']}_{data['it']}")
        t1 = time.time()
        df3 = df1.merge(df2, on=[0], algorithm='sort', env=env)
        env.barrier()
        t2 = time.time()
        t = (t2 - t1)
        #sum_t = comm.reduce(t)
        #tot_l = comm.reduce(len(df3))

        sum_t = communicator.allreduce(t, ReduceOp.SUM)

        tot_l = communicator.allreduce(len(df3), ReduceOp.SUM)

        if env.rank == 0:
            avg_t = sum_t / env.world_size
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l)
            StopWatch.stop(f"join_{i}_{data['host']}_{data['rows']}_{data['it']}")

    StopWatch.stop(f"join_total_{data['host']}_{data['rows']}_{data['it']}")

    if env.rank == 0:
        StopWatch.benchmark(tag=str(data))

    env.finalize()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="weak scaling")
    parser.add_argument('-n', dest='rows', type=int, required=True)
    parser.add_argument('-i', dest='it', type=int, default=10)
    parser.add_argument('-u', dest='unique', type=float, default=0.9, help="unique factor")
    parser.add_argument('-s', dest='scaling', type=str, default='w', choices=['s', 'w'],
                        help="s=strong w=weak")
    parser.add_argument('-w', dest='world_size', type=int, help="world size", required=True)

    parser.add_argument("-r", dest='redis_host', type=str, help="redis address",
                        required=True)

    parser.add_argument("-p1", dest='redis_port', type=int, help="name of redis port",
                        default=6379)  # 6379

    args = vars(parser.parse_args())
    args['host'] = "rivanna"
    for i in range(1):
        args['task'] = i
        # cylon_slice(args)
        cylon_join(args)
        #cylon_sort(args)

    # os.system(f"{git} branch | fgrep '*' ")
    # os.system(f"{git} rev-parse HEAD")
