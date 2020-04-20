from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string



cdef extern from "../../../cpp/src/twisterx/io/csv_read_config.h" namespace "twisterx::io::config":
    cdef cppclass _CSVReadOptions "twisterx::io::config::CSVReadOptions":
        _CSVReadOptions UseThreads(bool)
        _CSVReadOptions WithDelimiter(char)
        _CSVReadOptions IgnoreEmptyLines()
        _CSVReadOptions AutoGenerateColumnNames(vector[string])

