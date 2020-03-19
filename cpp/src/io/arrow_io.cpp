#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/csv/api.h>
#include "arrow_io.h"

arrow::Status read_csv(const std::string &path, std::shared_ptr<arrow::Table> table) {
  arrow::Status st;
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::InputStream> input;
  auto input_readable = std::dynamic_pointer_cast<arrow::io::ReadableFile>(input);
  arrow::Status status = arrow::io::ReadableFile::Open(path, pool, &input_readable);

  auto read_options = arrow::csv::ReadOptions::Defaults();
  auto parse_options = arrow::csv::ParseOptions::Defaults();
  auto convert_options = arrow::csv::ConvertOptions::Defaults();

  // Instantiate TableReader from input stream and options
  std::shared_ptr<arrow::csv::TableReader> reader;
  st = arrow::csv::TableReader::Make(pool, input, read_options,
                                     parse_options, convert_options,
                                     &reader);
  if (!st.ok()) {
    return st;
  }

  // Read table from CSV file
  st = reader->Read(&table);
  if (!st.ok()) {
    return st;
  }
}