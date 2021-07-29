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

'''
Run test:
>> pytest -q python/test/test_aggregate.py
'''

import numpy as np
import pyarrow as pa
import pandas as pd
import pycylon as cn
from pycylon import CylonContext


def test_aggregate():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    columns = 2

    data1 = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    data2 = np.array([10, 11, 12, 13, 14, 15], dtype=np.float32)

    nd_array_list = [data1, data2]

    ar_array: pa.array = pa.array(nd_array_list)

    ar_table: pa.Table = pa.Table.from_arrays(nd_array_list, names=['x0', 'x1'])

    ar1 = pa.array([1, 2, 3, 4])
    ar2 = pa.array(['a', 'b', 'c', 'd'])

    ar_tb2: pa.Table = pa.Table.from_arrays([ar1, ar2], names=['col1', 'col2'])

    assert isinstance(ar_tb2, pa.Table)

    col_names = ['col1', 'col2']

    cn_tb1 = cn.Table.from_numpy(ctx, col_names, nd_array_list)

    assert cn_tb1.row_count == data1.shape[0] and cn_tb1.column_count == len(nd_array_list)

    data_list = [[1, 2, 3, 4], ['p', 'q', 'r', 's']]

    cn_tb2 = cn.Table.from_list(ctx, col_names, data_list)

    assert cn_tb2.row_count == len(data_list[0]) and cn_tb2.column_count == len(data_list)

    dict1 = {'col1': [1, 2], 'col2': ['a', 'b']}

    ar_tb3: pa.Table = pa.Table.from_pydict(dict1)

    cn_tb3: cn.Table = cn.Table.from_pydict(ctx, dict1)

    assert cn_tb3.row_count == len(dict1['col1']) and cn_tb3.column_count == len(dict1)

    pdf = pd.DataFrame(dict1)

    # df, Schema schema=None, preserve_index=None, nthreads=None, columns=None, bool safe=True

    cn_tb4: cn.Table = cn.Table.from_pandas(ctx, pdf)

    assert cn_tb4.row_count == len(dict1['col1']) and cn_tb4.column_count == len(dict1)

    dict2 = {'col1': [1, 2, 3], 'col2': [2, 4, 3]}

    cn_tb5: cn.Table = cn.Table.from_pydict(ctx, dict2)

    assert cn_tb5.row_count == len(dict2['col1']) and cn_tb5.column_count == len(dict2)

    npy = cn_tb5.to_numpy()

    assert npy.shape == (len(dict2['col1']), len(dict2))

    dict3 = cn_tb5.to_pydict()

    assert dict3 == dict2

    for key1, key2 in zip(dict3.keys(), cn_tb5.column_names):
        assert key1 == key2

    assert cn_tb5.to_numpy().shape == (len(dict2['col1']), len(dict2))

    ## Aggregate Sum

    cn_tb6 = cn_tb5.sum('col1')

    assert cn_tb6.to_numpy()[0][0] == sum(dict2['col1'])

    cn_tb7 = cn_tb5.sum(0)

    assert cn_tb7.to_numpy()[0][0] == sum(dict2['col1'])

    ## Aggregate Count

    cn_tb8 = cn_tb5.count('col1')

    assert cn_tb8.to_numpy()[0][0] == len(dict2['col1'])

    cn_tb9 = cn_tb5.count(0)

    assert cn_tb9.to_numpy()[0][0] == len(dict2['col1'])

    ## Aggregate Min

    cn_tb10 = cn_tb5.min('col1')

    assert cn_tb10.to_numpy()[0][0] == min(dict2['col1'])

    cn_tb11 = cn_tb5.min(0)

    assert cn_tb11.to_numpy()[0][0] == min(dict2['col1'])

    ## Aggregate Max

    cn_tb12 = cn_tb5.max('col1')

    assert cn_tb12.to_numpy()[0][0] == max(dict2['col1'])

    cn_tb13 = cn_tb5.max(0)

    assert cn_tb13.to_numpy()[0][0] == max(dict2['col1'])

    from pycylon.data.aggregates import AggregationOp

    op1 = AggregationOp.SUM

    assert (op1 == AggregationOp.SUM)

    df = pd.DataFrame({'AnimalId': [1, 1, 2, 2, 3, 4, 4, 3],
                       'Max Speed': [380., 370., 24., 26., 23.1, 300.1, 310.2, 25.2]})

    ar_tb_gb = pa.Table.from_pandas(df)

    assert isinstance(ar_tb_gb, pa.Table)

    cn_tb_gb = cn.Table.from_arrow(ctx, ar_tb_gb)

    assert isinstance(cn_tb_gb, cn.Table)

    pdf1 = df.groupby(['AnimalId']).sum()

    cn_tb_gb_res = cn_tb_gb.groupby('AnimalId', {'Max Speed': AggregationOp.SUM}).sort(0)

    for val1, val2 in zip(cn_tb_gb_res.to_pydict()['sum_Max Speed'],
                          pdf1.to_dict()['Max Speed'].values()):
        assert val1 == val2

    cn_tb_gb_res1 = cn_tb_gb.groupby(0, {'Max Speed': 'sum'}).sort(0)

    for val1, val2 in zip(cn_tb_gb_res1.to_pydict()['sum_Max Speed'],
                          pdf1.to_dict()['Max Speed'].values()):
        assert val1 == val2

    pdf2 = df.groupby(['AnimalId']).min()

    cn_tb_gb_res2 = cn_tb_gb.groupby(0, {'Max Speed': 'min'}).sort(0)

    for val1, val2 in zip(cn_tb_gb_res2.to_pydict()['min_Max Speed'],
                          pdf2.to_dict()['Max Speed'].values()):
        assert val1 == val2

    pdf3 = df.groupby(['AnimalId']).max()

    cn_tb_gb_res3 = cn_tb_gb.groupby(0, {'Max Speed': AggregationOp.MAX}).sort(0)

    for val1, val2 in zip(cn_tb_gb_res3.to_pydict()['max_Max Speed'],
                          pdf3.to_dict()['Max Speed'].values()):
        assert val1 == val2


def test_aggregate_addons():
    ctx: CylonContext = CylonContext(config=None, distributed=False)

    from pycylon.data.aggregates import AggregationOp

    df_unq = pd.DataFrame({'AnimalId': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 3, 3, 4],
                           'AreaId': [21, 231, 211, 11, 12, 32, 42, 22, 23, 13, 44, 24, 34, 13, 13,
                                      41],
                           'Max Speed': [370., 370., 320, 320, 24., 26., 25., 24., 23.1, 23.1,
                                         300.1,
                                         310.2,
                                         310.2,
                                         25.2,
                                         25.2, 305.3],
                           'Avg Acceleration': [21, 21, 24, 11, 12, 32, 42, 22, 23, 13, 44, 24,
                                                34, 13, 13, 41],
                           'Avg Speed': [360., 330., 321, 310, 22., 23., 22., 21., 22.1, 21.1,
                                         300.0,
                                         305.2,
                                         303.2,
                                         25.0,
                                         25.1, 301.3]
                           })

    cn_tb_unq = cn.Table.from_pandas(ctx, df_unq)

    cn_tb_mul = cn_tb_unq.groupby(0, {'Max Speed': AggregationOp.NUNIQUE,
                                      'Avg Acceleration': AggregationOp.COUNT,
                                      'Avg Speed': [AggregationOp.MEAN, AggregationOp.VAR,
                                                    AggregationOp.STDDEV]}).sort(0)

    cn_tb_mul.set_index('AnimalId', drop=True)

    pdf_mul_grp = df_unq.groupby('AnimalId')

    pdf_mul = pdf_mul_grp.agg({'Max Speed': 'nunique', 'Avg Acceleration': 'count',
                               'Avg Speed': ['mean', 'var', 'std']})

    assert cn_tb_mul.index.index_values == list(pdf_mul_grp.groups.keys())

    # round values to 8 decimal places to fix floating point round issues
    assert np.array_equal(np.round(cn_tb_mul.to_pandas().values.tolist(), 8),
                          np.round(pdf_mul.values.tolist(), 8))
