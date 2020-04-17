#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/csv/api.h>
#include "arrow_io.hpp"

namespace twisterx {
    namespace io {
        arrow::Result<std::shared_ptr<arrow::Table>> read_csv(const std::string &path) {
            arrow::Status st;
            arrow::MemoryPool *pool = arrow::default_memory_pool();
            arrow::Result<std::shared_ptr<arrow::io::MemoryMappedFile>> mmap_result = arrow::io::MemoryMappedFile::Open(
                    path,
                    arrow::io::FileMode::READ);
            if (!mmap_result.status().ok()) {
                return mmap_result.status();
            }

            auto read_options = arrow::csv::ReadOptions::Defaults();
            auto parse_options = arrow::csv::ParseOptions::Defaults();
            auto convert_options = arrow::csv::ConvertOptions::Defaults();

            // Instantiate TableReader from input stream and options
            std::shared_ptr<arrow::csv::TableReader> reader = {};
            st = arrow::csv::TableReader::Make(pool, *mmap_result, read_options,
                                               parse_options, convert_options,
                                               &reader);
            if (!st.ok()) {
                return arrow::Result<std::shared_ptr<arrow::Table>>(st);
            }

            // Read table from CSV file
            //
            return reader->Read();
        }
    }
}