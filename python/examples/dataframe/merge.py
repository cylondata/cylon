from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig
import random

df1 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])
df2 = DataFrame([random.sample(range(10, 100), 50),
                 random.sample(range(10, 100), 50)])

# local merge
df3 = df1.merge(right=df2, on=[0])
print("Local Merge")
print(df3)

# distributed join
env = CylonEnv(config=MPIConfig())

print("Distributed Merge")
df3 = df1.merge(right=df2, on=[0], env=env)
print(df3)
