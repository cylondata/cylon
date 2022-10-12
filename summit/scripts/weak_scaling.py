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


# import os

# git = "/usr/bin/git"

def join(data=None):
    StopWatch.start(f"join_total_{data.host}_{data.n}_{data.it}")

    comm = MPI.COMM_WORLD

    config = MPIConfig(comm)
    env = CylonEnv(config=config, distributed=True)

    r = data.n
    u = data.u
    total_r = r * env.world_size

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(total_r * u), size=(r, 2))
    data2 = rng.integers(0, int(total_r * u), size=(r, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))

    for i in range(data.it):
        env.barrier()
        StopWatch.start(f"join_{i}_{data.host}_{data.n}_{data.it}")
        t1 = time.time()
        df3 = df1.merge(df2, on=[0], algorithm='sort', env=env)
        env.barrier()
        t2 = time.time()
        t = (t2 - t1) * 1000
        sum_t = comm.reduce(t)
        tot_l = comm.reduce(len(df3))

        if env.rank == 0:
            print("w", env.world_size, "r", r, "it", i, "t", sum_t / env.world_size, "l", tot_l)
            StopWatch.stop(f"join_{i}_{data.host}_{data.n}_{data.it}")

    StopWatch.stop(f"join_total_{data.host}_{data.n}_{data.it}")

    if env.rank == 0:
        StopWatch.benchmark(tag=str(data))

    env.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="weak scaling")
    parser.add_argument('-n', dest='rows', type=int, required=True)
    parser.add_argument('-i', dest='it', type=int, default=10)
    parser.add_argument('-u', dest='unique', type=float, default=0.9, help="unique factor")
    args = vars(parser.parse_args())

    data = dotdict()
    data.n = args['rows']
    data.it = args['it']
    data.u = args['unique']
    data.host = "summit"
    join(data)

    # os.system(f"{git} branch | fgrep '*' ")
    # os.system(f"{git} rev-parse HEAD")
