def test_additions_and_maps():
    from pycylon import Table
    from pycylon import CylonContext
    import pandas as pd
    import numpy as np

    ctx = CylonContext(config=None, distributed=False)

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
    from pycylon import Table
    from pycylon import CylonContext
    import pandas as pd
    import numpy as np

    ctx = CylonContext(config=None, distributed=False)

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
    from pycylon import Table
    from pycylon import CylonContext
    import pandas as pd
    import numpy as np

    ctx = CylonContext(config=None, distributed=False)

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
    from pycylon import Table
    from pycylon import CylonContext
    import pandas as pd
    import numpy as np

    ctx = CylonContext(config=None, distributed=False)

    pdf = pd.DataFrame(
        {'idx': ['x', 'y', 'z'], 'col-1': ["a", "b", "c"], 'col-2': [10, 20, 30], 'col-3': [
            'Y',
            'N',
            'Y']})

    tb = Table.from_pandas(ctx, pdf)

    assert tb.to_pydict(with_index=True) == pdf.to_dict()


def test_pdf_to_pdf_assign():
    import pandas as pd
    pdf1 = pd.DataFrame({'a': [1, 2, 3, 4, 5, 110, 111, 112, 113], 'b': [10, 11, 12, 13, 14, 5, 4,
                                                                         3, 2]}, index=[0, 1, 2,
                                                                                        3, 4, 5,
                                                                                        6, 7, 8])

    pdf2 = pd.DataFrame({'a': [10, 20, 30, 40, 50], 'b': [100, 101, 102, 103, 104]}, index=[0, 1,
                                                                                            2, 3,
                                                                                            4])

    pdf3 = pd.DataFrame({'a': [1, 2, 3, 4, 5, 110, 111, 112, 113], 'b': [1110, 1111, 1112, 1113,
                                                                         1114,
                                                                         115, 114,
                                                                         113, 112]}, index=[10, 11,
                                                                                            12, 13,
                                                                                            14, 15,
                                                                                            16, 17,
                                                                                            18])

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


