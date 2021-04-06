import pandas as pd
import numpy as np

from pycylon import DataFrame


def create_df(data):
    # np.T is a temp fix to address inconsistencies
    return DataFrame(data), pd.DataFrame(np.array(data).T)


def assert_eq(df_c: DataFrame, df_p: pd.DataFrame, sort=False):
    if sort:  # sort by every column
        print(df_c.sort_values(
            by=[*range(0, df_c.shape[1])]).to_numpy(order='F',  zero_copy_only=False))

        print(df_p.to_numpy())
        assert np.array_equal(df_c.sort_values(
            by=[*range(0, df_c.shape[1])]).to_pandas().to_numpy(),  df_p.sort_values(
            by=[*range(0, df_p.shape[1])]).to_numpy())
    else:
        assert np.array_equal(df_c.to_numpy(order='F',  zero_copy_only=False),  df_p.to_numpy())
