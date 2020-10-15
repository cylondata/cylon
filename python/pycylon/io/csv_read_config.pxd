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

'''
Mapping for TwisterX C++ CSV Read Config Options
(Currently we support majority of our data loading through PyArrow API)
'''

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.memory cimport shared_ptr

cdef extern from "../../../cpp/src/cylon/io/csv_read_config.hpp" namespace "cylon::io::config":
    cdef cppclass CCSVReadOptions "cylon::io::config::CSVReadOptions":
        CCSVReadOptions()

        CCSVReadOptions ConcurrentFileReads(bool concurrent_file_reads)

        bool IsConcurrentFileReads()

        CCSVReadOptions UseThreads(bool use_threads)

        CCSVReadOptions WithDelimiter(char delimiter)

        CCSVReadOptions IgnoreEmptyLines()

        CCSVReadOptions AutoGenerateColumnNames()

        CCSVReadOptions ColumnNames(const vector[string] &column_names)

        CCSVReadOptions BlockSize(int block_size)

        CCSVReadOptions UseQuoting()

        CCSVReadOptions WithQuoteChar(char quote_char)

        CCSVReadOptions DoubleQuote()

        CCSVReadOptions UseEscaping()

        CCSVReadOptions EscapingCharacter(char escaping_char)

        CCSVReadOptions HasNewLinesInValues()

        CCSVReadOptions SkipRows(int skip_rows)

        # NOTE: add upon requirement in Python
        # CCSVReadOptions WithColumnTypes(const unordered_map[string, shared_ptr[DataType]]
        #                                 &column_types)

        CCSVReadOptions NullValues(const vector[string] &null_value)

        CCSVReadOptions TrueValues(const vector[string] &true_values)

        CCSVReadOptions FalseValues(const vector[string] &false_values)

        CCSVReadOptions StringsCanBeNull()

        CCSVReadOptions IncludeColumns(const vector[string] &include_columns)

        CCSVReadOptions IncludeMissingColumns()

        shared_ptr[void] GetHolder() const


cdef class CSVReadOptions:
    cdef:
        CCSVReadOptions *thisPtr
        vector[string] cpp_strings






