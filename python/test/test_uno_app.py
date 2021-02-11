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
    pdf_idx_values = pdf.index.values

    print(tb_idx_values)
    print(pdf_idx_values)



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

    print(tb)

    import pyarrow as pa

    from operator import add

    print(tb)




test_str_ops()