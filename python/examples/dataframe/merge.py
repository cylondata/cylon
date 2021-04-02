from pycylon import DataFrame, CylonEnv
from pycylon.net import MPIConfig

df1 = DataFrame([[1, 2, 3], [2, 3, 4]])
df2 = DataFrame([[1, 1, 1], [2, 3, 4]])

# local merge
df3 = df1.merge(right=df2, on=[0, 1])
print("Local Merge")
print(df3)

# distributed join
env = CylonEnv(config=MPIConfig())

print("Distributed Merge")
df3 = df1.merge(right=df2, on=[0, 1], env=env)
print(df3)
