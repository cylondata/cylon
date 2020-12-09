from pycylon import Table
from pycylon import CylonContext

ctx: CylonContext = CylonContext(config=None, distributed=False)

data1 = [[False, True, False, True], [True, True, False, False]]
columns1 = ['col-1', 'col-2']

data2 = [[True, True, False, False], [False, True, False, True]]
columns2 = ['col-1', 'col-2']

tb1: Table = Table.from_list(ctx, columns1, data1)
tb2: Table = Table.from_list(ctx, columns2, data2)

print("Table 1")
print(tb1)

print("Table 2")
print(tb2)

tb_or = tb1 | tb2
print("Or")
print(tb_or)

tb_and = tb1 & tb2
print("And")
print(tb_and)

tb_inv = ~tb1
print("Table")
print(tb1)
print("Invert Table")
print(tb_inv)
