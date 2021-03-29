from pycylon import DataFrame


df1 = DataFrame([[1, 2, 3], [2, 3, 4]])
df2 = DataFrame([[1, 1, 1], [2, 3, 4]])
df3 = DataFrame([[8, 8, 10], [21, 31, 41]])


df3 = df3.drop_duplicates(subset=[0,1], keep='first')

print(df3)
