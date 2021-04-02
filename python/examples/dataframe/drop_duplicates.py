from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random

df1 = DataFrame([random.sample(range(0, 5), 5),
                 random.sample(range(0, 5), 5)])


# local unique
df3 = df1.drop_duplicates()
print("Local Unique")
print(df3)

# distributed unique
env = CylonEnv(config=MPIConfig())

print("Distributed Unique", env.rank)
df3 = df1.drop_duplicates(env=env)
print(df3)
