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


from typing import List
from pycylon.data.column import Column
from pycylon.data.data_type import DataType
import pyarrow as pa
import numpy as np
from collections.abc import Iterable
import pandas as pd


class Series(object):

    def __init__(self, series_id: str = None, data=None, data_type: DataType = None):
        if isinstance(data, List) or isinstance(data, np.ndarray):
            self._column = Column(series_id, data_type, pa.array(data))
        elif isinstance(data, pa.Array):
            self._column = Column(series_id, data_type, data)
        else:
            raise ValueError('Invalid data type, data must be List, Numpy NdArray or PyArrow array')

    @property
    def id(self):
        return self._column.id

    @property
    def data(self):
        return self._column.data

    @property
    def dtype(self):
        return self._column.dtype

    @property
    def shape(self):
        rows, cols = 0, 0
        d = self._column.data
        if isinstance(d, Iterable):
            if isinstance(d[0], Iterable):
                rows = len(d[0])
                cols = len(d)
            else:
                cols = 1
                rows = len(d)
            return cols, rows
        else:
            # when scalar is in the series
            # Not Yet supported in series.
            return 1

    def __getitem__(self, item):
        if isinstance(item, int) or isinstance(item, slice):
            return self._column.data[item]
        else:
            raise ValueError('Indexing in series is only based on numeric indices and slices')

    def __repr__(self):
        return pd.Series(self.data).__repr__()





