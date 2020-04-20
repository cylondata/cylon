import uuid
from pytwisterx.data import TableUtil
from pytwisterx.data import Table

tb_util = TableUtil
table_id: str = tb_util.read_csv('/tmp/csv.csv', ',')
print("From Table Utils, Table Id : {}".format(table_id))

tb = Table(table_id.encode())

print("From Python User, Table Id : {}".format(tb.id))

print("Table Columns : ", tb.columns)
print("Table Rows : ", tb.rows)

print("Table Show")
tb.show()





