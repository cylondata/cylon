from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random

df1 = DataFrame([random.sample(range(10, 30), 50),
                 random.sample(range(10, 30), 50)])


# local sort
df3 = df1.sort_values(by=[0])
print("Local Sort")
print(df3)

# distributed sort
env = CylonEnv(config=MPIConfig())

print("Distributed Sort", env.rank)
df3 = df1.sort_values(by=[0], env=env)
print(df3)
