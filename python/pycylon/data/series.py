from typing import List
from pycylon.data.column import Column
from pycylon.data.data_type import DataType
import pyarrow as pa


class Series(object):

    def __init__(self, series_id: str = None, data: List = None, data_type: DataType = None):
        self._column = Column(series_id, data_type, pa.array(data))

    @property
    def id(self):
        return self._column.id

    @property
    def data(self):
        return self._column.data

    @property
    def dtype(self):
        return self._column.dtype

    def __getitem__(self, item):
        if isinstance(item, int) or isinstance(item, slice):
            return self._column.data[item]
        else:
            raise ValueError('Indexing in series is only based on numeric indices and slices')





