/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "arrow_io.hpp"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/csv/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <memory>

#include "csv_read_config_holder.hpp"
#include "../ctx/cylon_context.hpp"
#include "../ctx/arrow_memory_pool_utils.hpp"

namespace cylon {
namespace io {

arrow::Result <std::shared_ptr<arrow::Table>> read_csv(std::shared_ptr<cylon::CylonContext> &ctx,
                                                       const std::string &path,
                                                       cylon::io::config::CSVReadOptions options) {
  arrow::Status st;
  auto *pool = cylon::ToArrowPool(ctx);
  arrow::Result <std::shared_ptr<arrow::io::MemoryMappedFile>> mmap_result =
      arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
  if (!mmap_result.status().ok()) {
    return mmap_result.status();
  }

  auto read_options = dynamic_cast<arrow::csv::ReadOptions *>(
      config::CSVConfigHolder::GetCastedHolder(options));
  auto parse_options = dynamic_cast<arrow::csv::ParseOptions *>(
      config::CSVConfigHolder::GetCastedHolder(options));
  auto convert_options = dynamic_cast<arrow::csv::ConvertOptions *>(
      config::CSVConfigHolder::GetCastedHolder(options));

  // Instantiate TableReader from input stream and options
  arrow::Result <std::shared_ptr<arrow::csv::TableReader>> reader =
      arrow::csv::TableReader::Make(pool, *mmap_result, *read_options,
                                    *parse_options, *convert_options);
  if (!reader.ok()) {
    return arrow::Result < std::shared_ptr < arrow::Table >> (reader.status());
  }

  // Read table from CSV file
  return (*reader)->Read();
}

// Read Parquet
arrow::Result <std::shared_ptr<arrow::Table>> read_parquet(std::shared_ptr<cylon::CylonContext> &ctx,
                                                           const std::string &path) {
    arrow::Status st;
    auto *pool = cylon::ToArrowPool(ctx);
    arrow::Result <std::shared_ptr<arrow::io::MemoryMappedFile>> mmap_result =
            arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);

    if (!mmap_result.status().ok()) {
        return mmap_result.status();
    }

    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    st = parquet::arrow::OpenFile(*mmap_result, pool, &arrow_reader);
    if (!st.ok()) {
        // Handle error instantiating file reader...
        return st;
    }

    // Read entire file as a single Arrow table
    std::shared_ptr<arrow::Table> table;
    st = arrow_reader->ReadTable(&table);
    if (!st.ok()) {
        // Handle error reading Parquet data...
        return st;
    }

    return table;
}

//Write Parquet
arrow::Status write_parquet(std::shared_ptr<arrow::Table> &table, const std::string &path){
    arrow::Result<std::shared_ptr<arrow::io::FileOutputStream>> outfile_result =
            arrow::io::FileOutputStream::Open(path);
    if (!outfile_result.status().ok()) {
        return outfile_result.status();
    }

    arrow::Status writefile_result =
            parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *outfile_result, 3);
    if (!writefile_result.ok()) {
        return writefile_result;
    }
    return (*outfile_result)->Close();
}

}  // namespace io
}  // namespace cylon
