from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random

df1 = DataFrame([[0, 0, 1, 1], [1, 10, 1, 5], [10, 20, 30, 40]])


df3 = df1.groupby(by=0).agg({
    "1": "sum",
    "2": "min"
})
print(df3)


df4 = df1.groupby(by=0).min()
print(df4)

df5 = df1.groupby(by=[0, 1]).max()
print(df5)
