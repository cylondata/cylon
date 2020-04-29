from pyarrow.lib cimport *


cdef class ArrowUtil:

    @staticmethod
    def get_array_length(obj):
        # Just an example function accessing both the pyarrow Cython API
        # and the Arrow C++ API
        cdef shared_ptr[CArray] arr = pyarrow_unwrap_array(obj)
        if arr.get() == NULL:
            raise TypeError("not an array")
        return arr.get().length()

    @staticmethod
    def get_table_info(obj):
        cdef shared_ptr[CTable] table = pyarrow_unwrap_table(obj)

        if table.get() == NULL:
            raise TypeError("not an table")
        else:
            print("OK")

