from pycylon.data.table import Table
from pycylon.index import Index, RangeIndex, NumericIndex

cdef class IndexEngine:

    def __cinit__(self, index: Index=None, n=0):
        """
        Initializes the IndexEngine
        Args:
            index: Index of the table which can be (Index, RangeIndex, NumericIndex)
            n: number of rows in the table. If index is not set, this will be used.
        """
        self._index_object = index
        self.n = n

    def get_loc(self, key):
        """
        Args:
            key: can be a row index/indices, row and column location [row_index, column_name],
            boolean condition/s, slice,
        Returns: A single value, a set of values or a Cylon Table
        """
        if self._index_object:
            if isinstance(self._index_object, RangeIndex):
                pass
            elif isinstance(self._index_object, NumericIndex):
                pass
            else:
                raise NotImplemented(f"The Index type {self._index_object} not supported.")
        elif self.n > 0:
            pass
        else:
            raise NotImplemented("Get location cannot be called because bad IndexEngine "
                                 "initialization. Initialize with Index or number of rows")

    def get_dis_loc(self,key):
        """
        Returns the element considering the distributed table by copying the data from the
        existing process to the called process. If a rank is not specified, will be returned to all
        processes.
        Args:
            key: index of value to be retrieved

        Returns: Cylon Table

        """
        return NotImplemented

