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

import os
import pyarrow as pa
from pyarrow import csv
from pycylon import Table
from pycylon.net import MPIConfig
from pycylon import CylonContext
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv

'''
running test case 
>>  pytest -q python/pycylon/test/test_pyarrow.py 
'''

def test_arrow_cylon():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    table_path = 'data/input/user_device_tm_1.csv'

    assert os.path.exists(table_path)

    tb: pa.Table = csv.read_csv(table_path)

    arrow_columns = len(tb.columns)
    arrow_rows = tb.num_rows

    tbc = Table.from_arrow(ctx, tb)

    cylon_rows = tbc.row_count
    cylon_columns = tbc.column_count

    assert arrow_columns == cylon_columns
    assert arrow_rows == cylon_rows

    ctx.finalize()