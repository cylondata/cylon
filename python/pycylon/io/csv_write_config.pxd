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