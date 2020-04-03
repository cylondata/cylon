from libcpp.string cimport string

from twisterx.table.table_api cimport table_api

cdef class PyTableAPI:

    @staticmethod
    def column_count() -> int:
        return 1;

    @staticmethod
    def row_count() -> int:
        return 1;
