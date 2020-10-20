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
Run test:
>> pytest -q python/test/test_table_properties.py
"""

import os
from pycylon import Table
from pycylon import CylonContext
from pycylon.io import CSVReadOptions
from pycylon.io import read_csv


# TODO: this test_class will be used to test the magic functions of Table API

def test_properties():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    table1_path = '/tmp/user_usage_tm_1.csv'
    table2_path = '/tmp/user_usage_tm_2.csv'

    assert os.path.exists(table1_path) and os.path.exists(table2_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb1: Table = read_csv(ctx, table1_path, csv_read_options)
    #
    # print("Table Column Names")
    # print(tb.column_names)
    #
    # print("Table Schema")
    # print(tb.schema)
    #
    # print(tb[0].to_pandas())
    #
    # print(tb[0:5].to_pandas())
    #
    # print(tb[2:5].to_pandas())
    #
    # print(tb[5].to_pandas())
    #
    # print(tb[7].to_pandas())
    #
    # tb.show_by_range(0, 4, 0, 4)
    #
    # print(tb[0:5].to_pandas())

    # ctx.finalize()

    # import pyarrow as pa

    # arw_table: pa.Table = tb.to_arrow()

