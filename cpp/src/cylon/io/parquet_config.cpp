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

#include <cstdint>

#include "parquet_config.hpp"

namespace cylon {
    namespace io {
        namespace config {
            ParquetOptions ParquetOptions::ConcurrentFileReads(bool file_reads) {
                this->concurrent_file_reads = file_reads;
                return *this;
            }
            bool ParquetOptions::IsConcurrentFileReads() {
                return this->concurrent_file_reads;
            }

            ParquetOptions ParquetOptions::ChunkSize(int64_t chunk_size_) {
                this->chunk_size = chunk_size_;
                return *this;
            }
            int64_t ParquetOptions::GetChunkSize() {
                return this->chunk_size;
            }

            ParquetOptions ParquetOptions::WriterProperties(std::shared_ptr<parquet::WriterProperties> &writer_properties_){
                this->writer_properties = writer_properties_;
                return *this;
            }
            std::shared_ptr<parquet::WriterProperties> ParquetOptions::GetWriterProperties() {
                return this->writer_properties;
            }

            ParquetOptions ParquetOptions::ArrowWriterProperties(
                    std::shared_ptr<parquet::ArrowWriterProperties> &arrow_writer_properties_){
                this->arrow_writer_properties = arrow_writer_properties_;
                return *this;
            }
            std::shared_ptr<parquet::ArrowWriterProperties> ParquetOptions::GetArrowWriterProperties() {
                return this->arrow_writer_properties;
            }

            ParquetOptions::ParquetOptions() {
            }
        }  // namespace config
    }  // namespace io
}  // namespace cylon
