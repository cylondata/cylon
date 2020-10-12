from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from pycylon.io.csv_write_config cimport CCSVWriteOptions

from typing import List


cdef class CSVWriteOptions:

    def __cinit__(self):
        self.thisPtr = new CCSVWriteOptions()

    def with_delimiter(self, delimiter) -> CSVWriteOptions:
        cdef string c_string = delimiter.encode()
        self.thisPtr.WithDelimiter(c_string[0])
        return self

    def with_column_names(self, column_names=[]):
        cdef vector[string] col_names = column_names
        self.thisPtr.ColumnNames(col_names)
        return self

    def delimiter(self) -> str:
        cdef char delm = self.thisPtr.GetDelimiter()
        return delm.decode()

    def column_names(self) -> List:
        cdef vector[string] col_names = self.thisPtr.GetColumnNames()
        return col_names

    def is_override_column_names(self) -> bool:
        return self.thisPtr.IsOverrideColumnNames()




