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

"""
Running the test

mpirun -n 2 python python/test/test_cylon_table_conversion.py --table1_path /tmp/csv1.csv --table2_path /tmp/csv2.csv
"""

from pycylon.csv import csv_reader
from pycylon import Table
from pycylon import CylonContext
import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='PyCylon Table Conversion')
parser.add_argument('--table1_path', type=str, help='Path to table 1 csv')
parser.add_argument('--table2_path', type=str, help='Path to table 2 csv')

args = parser.parse_args()

ctx: CylonContext = CylonContext("mpi")

tb1: Table = csv_reader.read(ctx, args.table1_path, ',')
tb2: Table = csv_reader.read(ctx, args.table2_path, ',')

tb1.show()

print("First Hello World From Rank {}, Size {}".format(ctx.get_rank(), ctx.get_world_size()))

tb3: Table = tb1.distributed_join(ctx, table=tb2, join_type='inner', algorithm='sort', left_col=0, right_col=0)

#pdf: pd.DataFrame = tb3.to_pandas()
npy: np.ndarray = tb3.to_numpy(order='C')

# Cylon table rows must be equal to the rows of pandas dataframe extracted from the table
#assert tb3.rows == pdf.shape[0]
# Cylon table columns must be equal to the columns of pandas dataframe extracted from the table
#assert tb3.columns == pdf.shape[1]
# Cylon table rows must be equal to the rows of numpy ndarray extracted from the table
assert tb3.rows == npy.shape[0]
# Cylon table columns must be equal to the columns of numpy ndarray extracted from the table
assert tb3.columns == npy.shape[1]

print(f"Rank[{ctx.get_rank()}]: Table.Rows={tb3.rows}, Table.Columns={tb3.columns}, "
      f"Numpy Array Shape = {npy.shape}")

print(f"Array Config Rank[{ctx.get_rank()}], {npy.flags} {npy.dtype}")

ctx.finalize()
