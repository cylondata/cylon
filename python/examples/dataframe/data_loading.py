from pycylon import DataFrame, read_csv, CylonEnv
from pycylon.net import MPIConfig
import sys
import pandas as pd


# using cylon native reader
df = read_csv(sys.argv[1])
print(df)

# using pandas to load csv
df = DataFrame(pd.read_csv(
    "http://data.un.org/_Docs/SYB/CSV/SYB63_1_202009_Population,%20Surface%20Area%20and%20Density.csv", skiprows=6, usecols=[0, 1]))
print(df)

# loading json
df = DataFrame(pd.read_json("https://api.exchangerate-api.com/v4/latest/USD"))
print(df)

# distributed loading : run in distributed mode with MPI or UCX
env = CylonEnv(config=MPIConfig())
df = read_csv(sys.argv[1], slice=True, env=env)
print(df)
