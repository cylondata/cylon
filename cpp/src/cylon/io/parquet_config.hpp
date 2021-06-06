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

#ifndef CYLON_SRC_CYLON_IO_PARQUET_CONFIG_HPP_
#define CYLON_SRC_CYLON_IO_PARQUET_CONFIG_HPP_

#include <cstdint>
#include <parquet/properties.h>

namespace cylon {
namespace io {
namespace config {
class ParquetOptions {

 private:
  int64_t chunk_size = 3;
  bool concurrent_file_reads = true;
  std::shared_ptr<parquet::WriterProperties> writer_properties = parquet::default_writer_properties();
  std::shared_ptr<parquet::ArrowWriterProperties> arrow_writer_properties =
      parquet::default_arrow_writer_properties();

 public:
  ParquetOptions();

  ParquetOptions ConcurrentFileReads(bool file_reads);
  bool IsConcurrentFileReads() const;

  ParquetOptions ChunkSize(int64_t chunk_size_);
  int64_t GetChunkSize() const;

  ParquetOptions WriterProperties(std::shared_ptr<parquet::WriterProperties> &writer_properties_);
  std::shared_ptr<parquet::WriterProperties> GetWriterProperties();

  ParquetOptions ArrowWriterProperties(std::shared_ptr<parquet::ArrowWriterProperties> &arrow_writer_properties_);
  std::shared_ptr<parquet::ArrowWriterProperties> GetArrowWriterProperties();

};

}  // namespace config
}  // namespace io
}  // namespace cylon
#endif //CYLON_SRC_CYLON_IO_PARQUET_CONFIG_HPP_
