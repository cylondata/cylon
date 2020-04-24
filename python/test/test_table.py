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

print("Write an already Loaded Table")

new_path: str = '/tmp/csv1.csv'
tb.to_csv(new_path)

tb2: Table = csv.read(new_path, ',')
tb2.show()

print("Joining Tables")

tb1: Table = csv.read('/tmp/csv.csv', ',')
tb2: Table = csv.read('/tmp/csv.csv', ',')
tb3: Table = tb2.join(table=tb1, join_type='inner', algorithm='sort', left_col=0, right_col=1)
print(tb3.id)
tb3.show()


ar: np.ndarray = tb3.to_numpy(cols=[0,1,12], dtype=np.float32)













