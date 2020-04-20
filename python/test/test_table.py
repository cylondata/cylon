import uuid
from pytwisterx.data import PyTable

table = PyTable

print(table.columns())
print(table.rows())
val = table.read_csv('/tmp/csv.csv', ',')

#print(val)




