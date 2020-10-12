from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool


cdef extern from "../../../cpp/src/cylon/io/csv_write_config.hpp" namespace "cylon::io::config":
    cdef cppclass CCSVWriteOptions "cylon::io::config::CSVWriteOptions":

        CCSVWriteOptions()

        CCSVWriteOptions WithDelimiter(char delimiter)

        CCSVWriteOptions ColumnNames(const vector[string] &column_names)

        char GetDelimiter() const

        vector[string] GetColumnNames() const

        bool IsOverrideColumnNames() const

cdef class CSVWriteOptions:
    cdef:
        CCSVWriteOptions *thisPtr