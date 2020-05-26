from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.memory cimport shared_ptr


cdef extern from "../../../cpp/src/twisterx/io/csv_read_config.h" namespace "twisterx::io::config":
    cdef cppclass _CSVReadOptions "twisterx::io::config::CSVReadOptions":
        _CSVReadOptions()

        _CSVReadOptions UseThreads(bool use_threads)

        _CSVReadOptions WithDelimiter(char delimiter)

        _CSVReadOptions IgnoreEmptyLines()

        _CSVReadOptions AutoGenerateColumnNames()

        _CSVReadOptions ColumnNames(const vector[string] &column_names);

        _CSVReadOptions BlockSize(int block_size);

        _CSVReadOptions UseQuoting();

        _CSVReadOptions WithQuoteChar(char quote_char);

        _CSVReadOptions DoubleQuote();

        _CSVReadOptions UseEscaping();

        _CSVReadOptions EscapingCharacter(char escaping_char);

        _CSVReadOptions HasNewLinesInValues();

        _CSVReadOptions SkipRows(int skip_rows);

        # std::unordered_map<std::string,std::shared_ptr<DataType>>
        #_CSVReadOptions WithColumnTypes(const unordered_map[string, shared_ptr[DataType]] &column_types);

        _CSVReadOptions NullValues(const vector[string] &null_value);

        _CSVReadOptions TrueValues(const vector[string] &true_values);

        _CSVReadOptions FalseValues(const vector[string] &false_values);

        _CSVReadOptions StringsCanBeNull();

        _CSVReadOptions IncludeColumns(const vector[string] &include_columns);

        _CSVReadOptions IncludeMissingColumns();








