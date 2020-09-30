from pycylon import Table
from pycylon.csv import csv_reader
from pycylon import CylonContext

ctx: CylonContext = CylonContext(config=None)

tb: Table = csv_reader.read(ctx, '/tmp/user_usage_tm_1.csv', ',')

print("Table Column Names")
print(tb.column_names)

print("Table Schema")
print(tb.schema)

print(tb[0].to_pandas())

print(tb[0:5].to_pandas())

print(tb[2:5].to_pandas())

print(tb[5].to_pandas())

print(tb[7].to_pandas())

tb.show_by_range(0, 4, 0, 4)

print(tb[0:5].to_pandas())

ctx.finalize()
