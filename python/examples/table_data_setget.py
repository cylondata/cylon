from pycylon import Table
from pycylon import CylonContext

ctx: CylonContext = CylonContext(config=None, distributed=False)

data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
columns = ['col-1', 'col-2', 'col-3']

tb: Table = Table.from_list(ctx, columns, data)
print("Full Dataset")
print(tb)

'''
Retrieve a subset of data as a Table by providing row range as a slice
'''
print("Retrieve via slicing")
tb1 = tb[1:3]

print(tb1)

'''
Retrieving a subset of data as a Table by providing column name/s
'''
print("Retrieve via column-name")
tb2 = tb['col-1']

print(tb2)

print("Retrieve via List of column-names")
tb3 = tb[['col-1', 'col-2']]

print(tb3)

'''
Retrieving a subset of data as a Table by providing a filter using a Table
'''

'''
Considering the full table
'''
print("Retrieve via a table filter")
tb4 = tb > 3  # this provides a bool Table

print(tb4)

tb5 = tb[tb4]

print(tb5)

tb6 = tb > 8  # this provides a bool Table

print(tb6)

tb7 = tb[tb6]

print(tb7)

'''
Considering a column
'''
print("Retrieve via a table-column filter")
tb8 = tb['col-1'] > 2

print(tb8)

tb9 = tb[tb8]

print(tb9)

'''
Set Values
'''

print(tb)

tb['col-3'] = Table.from_list(ctx, ['x'], [[90, 100, 110, 120]])

print(tb)

tb['col-4'] = Table.from_list(ctx, ['x'], [[190, 1100, 1110, 1120]])

print(tb)
