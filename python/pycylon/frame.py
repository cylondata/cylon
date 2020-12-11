from typing import List
from copy import copy
from collections.abc import Iterable
import pycylon as cn
import numpy as np
import pandas as pd
import pyarrow as pa
from pycylon import Series
from pycylon.index import RangeIndex, CategoricalIndex


class DataFrame(object):

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        self._index = None
        self._columns = []

    def _initialize_dataframe(self, data=None, index=None, columns=None, copy=False):
        rows = 0
        cols = 0
        self._table = None
        if isinstance(data, Iterable):
            if isinstance(data[0], Iterable):
                rows = len(data[0])
                cols = len(data)
            else:
                rows = len(data)
                cols = 1
            columns = self._initialize_columns(cols=cols, columns=columns)
            if copy:
                data = self._copy(data)
        elif isinstance(data, np.ndarray):
            rows, cols = data.shape
            columns = self._initialize_columns(cols=cols, columns=columns)
        elif isinstance(data, pd.DataFrame):
            rows, cols = data.shape
            columns = self._initialize_columns(cols=cols, columns=columns)
        elif isinstance(data, dict):
            cols = len(data)
            columns = self._initialize_columns(cols=cols, columns=columns)
            _, data_items = list(data.items())[0]
            rows = len(data_items)
        elif isinstance(data, pa.Table):
            rows, cols = data.shape
            columns = self._initialize_columns(cols=cols, columns=columns)
        elif isinstance(data, Series):
            cols, rows = data.shape
            columns = self._initialize_columns(cols=cols, columns=columns)
        else:
            raise ValueError(f"Invalid data structure, {type(data)}")

        if index is None:
            self._index = RangeIndex(start=0, stop=rows)
        else:
            if isinstance(index, CategoricalIndex):
                # check the validity of provided Index
                pass
            elif isinstance(index, RangeIndex):
                # check the validity of provided Index
                pass

    def _initialize_dtype(self, dtype):
        raise NotImplemented("Data type forcing is not implemented, only support inferring types")

    def _initialize_columns(self, cols, columns):
        if columns is None:
            return [str(i) for i in range(cols)]
        else:
            if isinstance(columns, Iterable):
                if len(columns) != cols:
                    raise ValueError(f"data columns count: {cols} and column names count "
                                     f"{len(columns)} not equal")
                else:
                    return columns

    def _copy(self, obj):
        return copy(obj)
