from pycylon import DataFrame, read_csv, CylonEnv
from pycylon.net import MPIConfig
import sys
import pandas as pd

# distributed loading : run in distributed mode with MPI or UCX
env = CylonEnv(config=MPIConfig())
df = read_csv(sys.argv[1], slice=True, env=env)
print(df)

env.finalize()
