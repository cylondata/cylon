from twisterx.io.csv_read_config cimport _CSVReadOptions
from libcpp cimport bool
from typing import List
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef class CSVReadOptions:
    cdef _CSVReadOptions *thisPtr
    cdef vector[string] cpp_strings

    def __cinit__(self):
        pass

    def __init__(self):
        pass

    def use_threads(self, use_thread: bool)-> CSVReadOptions:
        self.thisPtr.UseThreads(use_thread)
        return self

    def with_delimiter(self, delimiter: str) -> CSVReadOptions:
        cdef char c_delimiter = delimiter
        self.thisPtr.WithDelimiter(c_delimiter)
        return self

    def ignore_emptylines(self) -> CSVReadOptions:
        self.thisPtr.IgnoreEmptyLines()
        return self

    def auto_generate_column_names(self, column_names: List[str]) -> CSVReadOptions:
        pass

