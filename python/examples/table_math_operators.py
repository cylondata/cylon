from pycylon import Table
from pycylon import CylonContext

ctx: CylonContext = CylonContext(config=None, distributed=False)

data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
columns = ['col-1', 'col-2', 'col-3']

tb1: Table = Table.from_list(ctx, columns, data)

print(tb1)
scalar_value = 2

tb2 = -tb1
print("Negate")
print(tb2)


tb2 = tb1 + 2
print("Add")
print(tb2)

tb2 = tb1 - 2
print("Subtract")
print(tb2)

tb2 = tb1 * 2
print("Multiply")
print(tb2)

tb2 = tb1 / 2
print("Division")
print(tb2)

