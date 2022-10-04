import time

import pandas as pd
from mpi4py import MPI
from numpy.random import default_rng
from pycylon.frame import CylonEnv, DataFrame
from pycylon.net import MPIConfig


def join(r, it=4, u=0.9):
    comm = MPI.COMM_WORLD

    config = MPIConfig(comm)
    env = CylonEnv(config=config, distributed=True)

    total_r = r * env.world_size

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(total_r * u), size=(r, 2))
    data2 = rng.integers(0, int(total_r * u), size=(r, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))

    for i in range(it):
        env.barrier()
        t1 = time.time()
        df3 = df1.merge(df2, on=[0], algorithm='sort', env=env)
        env.barrier()
        t2 = time.time()
        t = (t2 - t1) * 1000
        sum_t = comm.reduce(t)
        tot_l = comm.reduce(len(df3))

        if env.rank == 0:
            print("w", env.world_size, "r", r, "it", i, "t", sum_t / env.world_size, "l", tot_l)

    env.finalize()


if __name__ == "__main__":
    join(1000000, it=10)
