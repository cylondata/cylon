import uuid
from pytwisterx.data import csv
from pytwisterx.data import Table

tb: Table = csv.read('/tmp/csv.csv', ',')

print("From Python User, Table Id : {}".format(tb.id))

print("Table Columns : ", tb.columns)
print("Table Rows : ", tb.rows)

print("Table Show")
tb.show()

print('Table By Range')
tb.show_by_range(0,2,0,2)








