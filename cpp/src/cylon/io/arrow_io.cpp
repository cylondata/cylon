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

#include <arrow/io/api.h>
#include <arrow/csv/api.h>
#include <memory>

#ifdef BUILD_CYLON_PARQUET
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#endif

#include "csv_read_config_holder.hpp"

namespace cylon {
namespace io {

arrow::Result<std::shared_ptr<arrow::Table>> read_csv(std::shared_ptr<cylon::CylonContext> &ctx,
                                                      const std::string &path,
                                                      cylon::io::config::CSVReadOptions options) {
  arrow::Status st;
  auto *pool = cylon::ToArrowPool(ctx);
  const auto &mmap_result = arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
  if (!mmap_result.ok()) {
    return mmap_result.status();
  }

  auto *read_options = dynamic_cast<arrow::csv::ReadOptions *>(
      config::CSVConfigHolder::GetCastedHolder(options));
  auto parse_options = dynamic_cast<arrow::csv::ParseOptions *>(
      config::CSVConfigHolder::GetCastedHolder(options));
  auto convert_options = dynamic_cast<arrow::csv::ConvertOptions *>(
      config::CSVConfigHolder::GetCastedHolder(options));

  // Instantiate TableReader from input stream and options
  arrow::io::IOContext io_ctx(pool);
  const auto &reader = arrow::csv::TableReader::Make(std::move(io_ctx), mmap_result.ValueOrDie(), *read_options,
                                                     *parse_options, *convert_options);
  if (!reader.ok()) {
    return reader.status();
  }

  // Read table from CSV file
  return (*reader)->Read();
}

#ifdef BUILD_CYLON_PARQUET
// Read Parquet
arrow::Result<std::shared_ptr<arrow::Table>> ReadParquet(std::shared_ptr<cylon::CylonContext> &ctx,
                                                         const std::string &path) {
  arrow::Status st;
  auto *pool = cylon::ToArrowPool(ctx);
  arrow::Result<std::shared_ptr<arrow::io::MemoryMappedFile>> mmapResult =
      arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::READ);
  if (!mmapResult.status().ok()) {
    return mmapResult.status();
  }

  std::unique_ptr<parquet::arrow::FileReader> arrowReader;
  st = parquet::arrow::OpenFile(mmapResult.ValueOrDie(), pool, &arrowReader);
  if (!st.ok()) {
    // Handle error instantiating file reader...
    return st;
  }

  // Read entire file as a single Arrow table
  std::shared_ptr<arrow::Table> table;
  st = arrowReader->ReadTable(&table);
  if (!st.ok()) {
    // Handle error reading Parquet data...
    return st;
  }

  return table;
}

//Write Parquet
arrow::Status WriteParquet(std::shared_ptr<cylon::CylonContext> &ctx,
                           std::shared_ptr<cylon::Table> &table,
                           const std::string &path,
                           cylon::io::config::ParquetOptions options) {
  auto *pool = cylon::ToArrowPool(ctx);
  arrow::Result<std::shared_ptr<arrow::io::FileOutputStream>> outfileResult =
      arrow::io::FileOutputStream::Open(path);
  if (!outfileResult.status().ok()) {
    return outfileResult.status();
  }

  arrow::Status writefileResult =
      parquet::arrow::WriteTable(*table->get_table(),
                                 pool,
                                 *outfileResult,
                                 options.GetChunkSize(),
                                 options.GetWriterProperties(),
                                 options.GetArrowWriterProperties());
  if (!writefileResult.ok()) {
    return writefileResult;
  }
  return (*outfileResult)->Close();
}
#endif

}  // namespace io
}  // namespace cylon
