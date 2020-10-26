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
from pandas import DataFrame
import pyarrow as pa

'''
Run test:
>> pytest -q python/test/test_table_properties.py
'''


def test_properties():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    table1_path = '/tmp/user_usage_tm_1.csv'
    table2_path = '/tmp/user_usage_tm_2.csv'

    assert os.path.exists(table1_path) and os.path.exists(table2_path)

    csv_read_options = CSVReadOptions().use_threads(True).block_size(1 << 30)

    tb: Table = read_csv(ctx, table1_path, csv_read_options)

    pdf = tb.to_pandas()

    # Test Columb based filter on EQ
    tb_filter = tb["monthly_mb"] == 519.12

    pdf_filter = pdf["monthly_mb"] == 519.12

    filtered_pdf = pdf[pdf_filter]

    filtered_tb = tb[tb_filter]

    tb_filter_pdf = tb_filter.to_pandas()

    filtered_tb_pdf = filtered_tb.to_pandas()

    assert tb_filter_pdf.values.flatten().tolist() == pdf_filter.values.flatten().tolist()

    assert filtered_tb_pdf.values.tolist() == filtered_pdf.values.tolist()

    # Test Table based filter on EQ

    tb_filter_all = tb == 519.12
    all_filtered_tb = tb[tb_filter_all]

    pdf_filter_all = pdf == 519.12
    all_filtered_pdf = pdf[pdf_filter_all]

    tb_filter_all_pdf = tb_filter_all.to_pandas()

    assert tb_filter_all_pdf.values.tolist() == pdf_filter_all.values.tolist()

    all_filtered_tb_pdf = all_filtered_tb.to_pandas()

    assert all_filtered_tb_pdf.fillna(0).values.tolist() == all_filtered_pdf.fillna(
        0).values.tolist()


test_properties()
