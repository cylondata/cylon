from pycylon.data.table import Table
from pycylon.index import Index, RangeIndex, NumericIndex, ColumnIndex, CategoricalIndex

cdef class _LocIndexer:

    def __cinit__(self, data_object: Table=None):
        """
        Initializes the IndexEngine
        Args:
            data_object: table which contain index of types (Index, RangeIndex, NumericIndex,
            CategoricalIndex, ColumnIndex
            )
        """
        self._data_object = data_object

    def get_loc(self, key):
        """
        Args:
            key: can be a row index/indices, row and column location [row_index, column_name],
            boolean condition/s, slice,
        Returns: A single value, a set of values or a Cylon Table
        """
        pass


    def get_dis_loc(self,key):
        """
        Returns the element considering the distributed table by copying the data from the
        existing process to the called process. If a rank is not specified, will be returned to all
        processes.
        Args:
            key: index of value to be retrieved

        Returns: Cylon Table

        """
        pass


cdef class LocIndexr(_LocIndexer):

    def __cinit__(self, data_object: Table=None):
        super().__cinit__(data_object)

    def get_loc(self, key):
        index_object = self._data_object.index
        if index_object:
            if isinstance(index_object, RangeIndex):
                pass
            elif isinstance(index_object, NumericIndex):
                pass
            else:
                raise NotImplemented(f"The Index type {index_object} not supported.")
        else:
            pass

    def get_dis_loc(self,key):
        return NotImplemented

