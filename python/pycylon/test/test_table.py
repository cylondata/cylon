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

>>> pytest -q python/pycylon/test/test_table.py
"""

from pyarrow.csv import read_csv as pyarrow_read_csv
from pycylon import Table
from pycylon import CylonContext
from pycylon.io import CSVReadOptions
from pycylon.io import CSVWriteOptions
from pycylon.io import read_csv
import pyarrow as pa


def test_table():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    table_path = 'data/input/user_device_tm_1.csv'

    pyarrow_table = pyarrow_read_csv(table_path)

    tb = Table(pyarrow_table, ctx)

    assert isinstance(tb, Table)

    ar_tb2 = tb.to_arrow()

    assert isinstance(ar_tb2, pa.Table)

    tb2 = Table.from_arrow(ctx, pyarrow_table)

    assert tb2.row_count == 272 and tb2.column_count == 4

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb3 = read_csv(ctx, table_path, csv_read_options)

    assert tb3.row_count == 272 and tb3.column_count == 4

    csv_write_options = CSVWriteOptions().with_delimiter(',')

    tb3.to_csv('/tmp/temp_record.csv', csv_write_options)

    tb4 = tb3.sort(1)

    col_names = ['use_id', 'user_id', 'platform_version', 'use_type_id']

    for idx, col in enumerate(col_names):
        assert tb4.column_names[idx] == col

    assert tb4.row_count == 272 and tb4.column_count == 4

    tb5 = tb3.sort('use_type_id')

    assert tb5.row_count == 272 and tb5.column_count == 4

    for idx, col in enumerate(col_names):
        assert tb5.column_names[idx] == col

    tb6 = Table.merge([tb4, tb4])

    assert tb6.row_count == 544 and tb6.column_count == 4

    tb7 = tb6

    assert tb7.row_count == 544 and tb7.column_count == 4

    tb8 = tb3.project([0, 1])

    assert tb8.row_count == 272 and tb8.column_count == 2

    tb9 = tb3.project(['use_id', 'platform_version'])

    assert tb9.row_count == 272 and tb9.column_count == 2

    project_col_names = ['use_id', 'platform_version']

    for idx, col in enumerate(project_col_names):
        assert tb9.column_names[idx] == col

    ctx.finalize()
