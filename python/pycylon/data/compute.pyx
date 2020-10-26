from pycylon.data.table cimport CTable
from pycylon.data.table import Table



cdef api c_filter(tb: Table, op):
    print(tb, op)