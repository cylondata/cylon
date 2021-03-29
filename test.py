from pycylon import DataFrame


df1 = DataFrame([[1, 2, 3], [2, 3, 4]])
df2 = DataFrame([[1, 1, 1], [2, 3, 4]])
df3 = DataFrame([[8, 9, 10], [21, 31, 41]])


joined = DataFrame.concat(objs=[df1,df2, df3])

print(joined)
