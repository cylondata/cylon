from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

df1 = DataFrame([[1, 2, 3], [2, 3, 4]])
df2 = DataFrame([[1, 1, 1], [2, 3, 4]])
df2.set_index([0])


# local join
df3 = df1.join(other=df2, on=[0])
print("Local Join")
print(df3)

# distributed join
env = CylonEnv(config=MPIConfig())

print("Distributed Join")
df3 = df1.join(other=df2, on=[0], env=env)
print(df3)

