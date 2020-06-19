##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##

from pytwisterx.data.table import csv_reader
from pytwisterx.data.table import Table
from pyarrow import Table as PyArrowTable
import time
from pytwisterx.ctx.context import TwisterxContext

ctx: TwisterxContext = TwisterxContext("mpi")

print('Loading Simple CSV File with Twisterx APIs')
print("----------------------------------------------------")

tb: Table = csv_reader.read(ctx, '/tmp/csv.csv', ',')
print("----------------------------------------------------")
print("From Python User, Table Id : {}".format(tb.id))
print("Table Columns : ", tb.columns)
print("Table Rows : ", tb.rows)
print("Table Show")
print("----------------------------------------------------")
tb.show()

print('Table By Range')
print("----------------------------------------------------")
tb.show_by_range(0,2,0,2)

print("Write an already Loaded Table")
print("----------------------------------------------------")
new_path: str = '/tmp/csv1.csv'
tb.to_csv(new_path)

tb2: Table = csv_reader.read(ctx, new_path, ',')
tb2.show()

print("Joining Tables")
print("----------------------------------------------------")
tb1: Table = csv_reader.read(ctx, '/tmp/csv.csv', ',')
tb2: Table = csv_reader.read(ctx, '/tmp/csv.csv', ',')
tb3: Table = tb2.join(ctx, table=tb1, join_type='inner', algorithm='sort', left_col=0, right_col=1)
print(tb3.id)
tb3.show()

print("===============================================================================================================")
py_arrow_csv_loading_time: int = 0
pyarrow_tb_to_tx_table_time: int = 0
tx_join_time: int = 0
tx_table_to_pyarrow_tb: int = 0

from pyarrow import csv

fn: str = '/tmp/csv.csv'
t1 = time.time_ns()
table = csv.read_csv(fn)
py_arrow_csv_loading_time = time.time_ns() - t1
#print(table)
#print(type(table))
#df = table.to_pandas()

t1 = time.time_ns()
table_frm_arrow: Table = Table.from_arrow(table)
pyarrow_tb_to_tx_table_time = time.time_ns() - t1

#table_frm_arrow.show()

print("Joining Loaded table from Python with Twisterx APIs")

t1 = time.time_ns()
tb4: Table = table_frm_arrow.join(ctx, table=table_frm_arrow, join_type='inner', algorithm='sort', left_col=0, right_col=0)
tx_join_time = time.time_ns() - t1

print("Result")

#tb4.show()
t1 = time.time_ns()
tbx: PyArrowTable = Table.to_arrow(tb4)
tx_table_to_pyarrow_tb = time.time_ns() - t1
#print(tbx)
dfx = tbx.to_pandas()
#print(dfx)
npr = dfx.to_numpy()

print(npr.shape)


print("Stats")
print("---------------------------------------------------------------------------------------------------------------")
print("PyArrow CSV Load Time : {} ms".format(py_arrow_csv_loading_time / 1000000))
print("PyArrow Table to Twisterx Table Conversion Time : {} ms".format(pyarrow_tb_to_tx_table_time / 1000000))
print("Twisterx Table Join Time: {} ms".format(tx_join_time / 1000000))
print("Twiterx Table to PyArrow Table Conversion Time : {} ms".format(tx_table_to_pyarrow_tb / 1000000))

# spark_pandas_to_dataframe : 38108.9990289 ms, tx-time 127 ms
# spark join : 261.089 ms, tx-time 176.189 (sort + join)






