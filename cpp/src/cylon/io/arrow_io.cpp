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

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/csv/api.h>

#include "arrow_io.hpp"
#include "csv_read_config_holder.hpp"
#include "../ctx/cylon_context.hpp"
#include "../ctx/arrow_memory_pool_utils.hpp"

namespace cylon {
namespace io {
arrow::Result<std::shared_ptr<arrow::Table>> read_csv(cylon::CylonContext *ctx,
                                                      const std::string &path,
                                                      cylon::io::config::CSVReadOptions options) {
  arrow::Status st;
  auto *pool = cylon::ToArrowPool(ctx);
  arrow::Result<std::shared_ptr<arrow::io::MemoryMappedFile>> mmap_result = arrow::io::MemoryMappedFile::Open(path,
                                                                                                              arrow::io::FileMode::READ);
  if (!mmap_result.status().ok()) {
    return mmap_result.status();
  }

  arrow::Array

  auto read_options = dynamic_cast<arrow::csv::ReadOptions *>(config::CSVConfigHolder::GetCastedHolder(options));
  auto parse_options = dynamic_cast<arrow::csv::ParseOptions *>(config::CSVConfigHolder::GetCastedHolder(options));
  auto convert_options = dynamic_cast<arrow::csv::ConvertOptions *>(config::CSVConfigHolder::GetCastedHolder(options));

  // Instantiate TableReader from input stream and options
  std::shared_ptr<arrow::csv::TableReader> reader;
  st = arrow::csv::TableReader::Make(pool, *mmap_result, *read_options,
                                     *parse_options, *convert_options,
                                     &reader);
  if (!st.ok()) {
    return arrow::Result<std::shared_ptr<arrow::Table>>(st);
  }

  // Read table from CSV file
  return reader->Read();
}
}
}