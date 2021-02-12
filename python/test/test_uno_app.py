from pycylon import Table
from pycylon import CylonContext
from pycylon.data.aggregates import AggregationOp
import pandas as pd
import numpy as np

ctx = CylonContext(config=None, distributed=False)


def test_additions_and_maps():
    from pycylon import Table
    from pycylon import CylonContext
    import pandas as pd
    import numpy as np

    pdf = pd.DataFrame(
        {'idx': ['x', 'y', 'z'], 'col-1': ["a", "b", "c"], 'col-2': [10, 20, 30], 'col-3': [
            'Y',
            'N',
            'Y']})

    tb = Table.from_pandas(ctx, pdf)

    print(tb)

    tb_s = tb['col-1'].applymap(lambda x: x + "_i")
    tb_log = tb['col-2'].applymap(lambda x: np.log10(x))
    tb_y = tb['col-3'].applymap(lambda x: (x == 'Y'))

    tb['col-1'] = tb_s
    tb['col-2'] = tb_log

    tb = tb[tb_y]
    pdf = pdf[pdf['col-3'].map(lambda x: (x == 'Y'))]

    print(pdf.to_dict())

    print(tb.to_pydict())


def test_default_indexing():
    pdf = pd.DataFrame(
        {'idx': ['x', 'y', 'z'], 'col-1': ["a", "b", "c"], 'col-2': [10, 20, 30], 'col-3': [
            'Y',
            'N',
            'Y']})

    tb = Table.from_pandas(ctx, pdf)

    tb_idx_values = tb.index.index_values
    pdf_idx_values = pdf.index.values.tolist()

    assert tb_idx_values == pdf_idx_values


def test_str_ops():
    pdf = pd.DataFrame(
        {'idx': ['x', 'y', 'z'], 'col-1': ["a", "b", "c"], 'col-2': [10, 20, 30], 'col-3': [
            'Y',
            'N',
            'Y']})

    tb = Table.from_pandas(ctx, pdf)

    pdf_str_val = pdf['col-1'] + "_" + pdf['col-3']
    tb_str_val = tb['col-1'] + "_" + tb['col-3']

    assert pdf_str_val.values.tolist() == tb_str_val.to_pandas().values.flatten().tolist()


def test_tb_to_pydict_with_index():
    pdf = pd.DataFrame(
        {'idx': ['x', 'y', 'z'], 'col-1': ["a", "b", "c"], 'col-2': [10, 20, 30], 'col-3': [
            'Y',
            'N',
            'Y']})

    tb = Table.from_pandas(ctx, pdf)

    assert tb.to_pydict(with_index=True) == pdf.to_dict()


def test_pdf_to_pdf_assign():
    pdf1 = pd.DataFrame({'a': [1, 2, 3, 4, 5, 110, 111, 112, 113], 'b': [10, 11, 12, 13, 14, 5, 4,
                                                                         3, 2]}, index=[0, 1, 2,
                                                                                        3, 4, 5,
                                                                                        6, 7, 8])

    pdf2 = pd.DataFrame({'a': [10, 20, 30, 40, 50], 'b': [100, 101, 102, 103, 104]}, index=[0, 1,
                                                                                            2, 3,
                                                                                            4])

    pdf3 = pd.DataFrame({'a': [1, 2, 3, 4, 5, 110, 111, 112, 113], 'b': [1110, 1111, 1112, 1113,
                                                                         1114, 115, 114, 113, 112]},
                        index=[10, 11, 12, 13, 14, 15, 16, 17, 18])

    tb1 = Table.from_pandas(ctx, pdf1)
    tb2 = Table.from_pandas(ctx, pdf2)
    tb3 = Table.from_pandas(ctx, pdf3)

    print(pdf1)
    print("-----------")
    print(pdf2)
    print("-----------")
    gp = pdf1['b']
    # print(pdf1['b'] < 6)
    print(gp[pdf1['b'] < 6])
    print(gp)
    print("-----------")
    gp[pdf1['b'] < 6] = pdf3['b']
    print(gp)


def test_groupby_with_indexing():
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

    cn_tb_unq = Table.from_pandas(ctx, df_unq)

    cn_tb_mul = cn_tb_unq.groupby('AnimalId', ['Max Speed', 'Avg Acceleration', 'Avg Speed'],
                                  [AggregationOp.NUNIQUE, AggregationOp.COUNT,
                                   AggregationOp.MEAN]).sort(0)

    print(cn_tb_mul)

    cn_tb_mul.set_index('AnimalId', drop=True)

    pdf_mul_grp = df_unq.groupby('AnimalId')

    pdf_mul = pdf_mul_grp.agg({'Max Speed': 'nunique', 'Avg Acceleration':
        'count', 'Avg Speed': 'mean'})

    print(pdf_mul_grp.groups.keys())
    print(pdf_mul.index)

    print(pdf_mul)


test_pdf_to_pdf_assign()
test_groupby_with_indexing()