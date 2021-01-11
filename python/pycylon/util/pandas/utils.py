from typing import List
import pandas as pd


def rename_with_new_column_names(df: pd.DataFrame, new_columns: List[str]):
    old_names = df.columns.array
    rename_map = {}
    for old_name, new_name in zip(old_names, new_columns):
        rename_map[old_name] = new_name
    df.rename(columns=rename_map, inplace=True)
    return df