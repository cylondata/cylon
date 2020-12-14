from pycylon import Table
from pycylon import CylonContext

ctx: CylonContext = CylonContext(config=None, distributed=False)

data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
columns = ['col-1', 'col-2', 'col-3']

tb: Table = Table.from_list(ctx, columns, data)
print("Full Dataset")
print(tb)

print("Equal Operator")

tb1 = tb['col-1'] == 2

print(tb1)

tb2 = tb == 2

print(tb2)


print("Inequal Operator")

tb3 = tb['col-1'] != 2

print(tb3)

tb4 = tb != 2

print(tb4)

print("Lesser Operator")

tb3 = tb['col-1'] < 2

print(tb3)

tb4 = tb < 2

print(tb4)

print("Lesser equal Operator")

tb3 = tb['col-1'] <= 2

print(tb3)

tb4 = tb <= 2

print(tb4)

print("Greater Operator")

tb3 = tb['col-1'] > 2

print(tb3)

tb4 = tb > 2

print(tb4)

print("Greater equal Operator")

tb3 = tb['col-1'] >= 2

print(tb3)

tb4 = tb >= 2

print(tb4)