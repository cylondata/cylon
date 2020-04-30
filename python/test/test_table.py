from pytwisterx.data import csv_reader
from pytwisterx.data import Table

tb: Table = csv_reader.read('/tmp/csv.csv', ',')

print("From Python User, Table Id : {}".format(tb.id))

print("Table Columns : ", tb.columns)
print("Table Rows : ", tb.rows)

print("Table Show")
tb.show()

print('Table By Range')
tb.show_by_range(0,2,0,2)

print("Write an already Loaded Table")

new_path: str = '/tmp/csv1.csv'
tb.to_csv(new_path)

tb2: Table = csv_reader.read(new_path, ',')
tb2.show()

print("Joining Tables")

tb1: Table = csv_reader.read('/tmp/csv.csv', ',')
tb2: Table = csv_reader.read('/tmp/csv.csv', ',')
tb3: Table = tb2.join(table=tb1, join_type='inner', algorithm='sort', left_col=0, right_col=1)
print(tb3.id)
tb3.show()

print("===============================")

# from pyarrow import csv
#
# fn = '/tmp/csv.csv'
#
# table = csv.read_csv(fn)
#
# print(table)









