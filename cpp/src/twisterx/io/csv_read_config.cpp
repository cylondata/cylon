#include "csv_read_config.h"


namespace twisterx {
    namespace io {
        namespace config {
            CSVReadOptions twisterx::io::config::CSVReadOptions::UseThreads(bool use_threads) {
                this->use_threads = use_threads;
                return *this;
            }

            CSVReadOptions twisterx::io::config::CSVReadOptions::WithDelimiter(char delimiter) {
                this->delimiter = delimiter;
                return *this;
            }

            CSVReadOptions twisterx::io::config::CSVReadOptions::IgnoreEmptyLines() {
                this->ignore_empty_lines = true;
                return *this;
            }

            CSVReadOptions twisterx::io::config::CSVReadOptions::AutoGenerateColumnNames() {
                this->autogenerate_column_names = true;
                return *this;
            }

            CSVReadOptions twisterx::io::config::CSVReadOptions::ColumnNames(std::vector<std::string> column_names) {
                this->column_names = column_names;
                return *this;
            }
        }
    }
}
