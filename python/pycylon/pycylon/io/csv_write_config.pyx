##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from pycylon.io.csv_write_config cimport CCSVWriteOptions

from typing import List


cdef class CSVWriteOptions:

    def __cinit__(self):
        self.thisPtr = new CCSVWriteOptions()

    def __del__(self):
        del self.thisPtr

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




