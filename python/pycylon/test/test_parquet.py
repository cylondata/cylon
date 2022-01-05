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
>> pytest -q python/pycylon/test/test_parquet.py
"""
import pandas as pd
from pyarrow.parquet import read_table
from pycylon.frame import DataFrame, CylonEnv


def test_read_parquet():
    tbl = read_table('data/input/parquet1_0.parquet')
    cdf = DataFrame(tbl)
    pdf = pd.read_parquet('file://data/input/parquet1_0.parquet')

    assert (pdf.values.tolist() == cdf.to_pandas().values.tolist())


def test_parquet_join():
    cdf1 = DataFrame(read_table('data/input/parquet1_0.parquet'))
    cdf2 = DataFrame(read_table('data/input/parquet2_0.parquet'))
    expected = DataFrame(read_table('data/output/join_inner_1_0.parquet'))

    out = cdf1.merge(cdf2, how='inner', on=[0], algorithm='sort', suffixes=('lt-', 'rt-'))

    assert(expected.equals(out, ordered=False))
    assert (len(expected.to_table().subtract(out.to_table())) == 0)
