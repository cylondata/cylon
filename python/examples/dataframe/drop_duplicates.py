from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random

df1 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])

# local unique
df3 = df1.drop_duplicates()
print("Local Unique")
print(df3)

# distributed unique
env = CylonEnv(config=MPIConfig())

df1 = DataFrame([random.sample(range(10*env.rank, 15*(env.rank+1)), 5),
                 random.sample(range(10*env.rank, 15*(env.rank+1)), 5)])
print("Distributed Unique", env.rank)
df3 = df1.drop_duplicates(env=env)
print(df3)

env.finalize()