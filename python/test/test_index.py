import pandas as pd
from pycylon.index import Index, RangeIndex, NumericIndex


def test_with_pandas():
    pdf = pd.DataFrame([[1, 2, 3, 4, 5, 'a'], [6, 7, 8, 9, 10, 'b'], [11, 12, 13, 14, 15, 'c'],
                        [16, 17, 18, 19, 20, 'a'], [16, 17, 18, 19, 20, 'd'],
                        [111, 112, 113, 114, 5,
                         'a']])

    # print(pdf)
    # pdf1 = pdf.set_index([1,2])
    # print(pdf1)
    # print(pdf1.index)


def test_numeric_index():
    rg = range(0, 10, 1)
    rg1 = range(0, 10, 2)
    r = NumericIndex(data=rg)

    assert r.index_values == rg
    assert r.index_values != rg1


def test_range_index():
    rg = range(0, 10, 1)
    rg1 = range(0, 10, 2)
    r = RangeIndex(start=rg.start, stop=rg.stop, step=rg.step)

    assert r.index_values == rg
    assert r.index_values != rg1



test_range_index()
