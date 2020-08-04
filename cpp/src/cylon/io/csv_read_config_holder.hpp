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

#ifndef CYLON_SRC_CYLON_IO_CSV_READ_CONFIG_HOLDER_HPP_
#define CYLON_SRC_CYLON_IO_CSV_READ_CONFIG_HOLDER_HPP_
#include <arrow/csv/options.h>
#include "csv_read_config.hpp"

namespace cylon {
namespace io {
namespace config {
/**
 * This is a helper class to hold the arrow CSV read options.
 * This class shouldn't be used with other language interfaces, due to the arrow
 * dependency.
 */
class CSVConfigHolder : public arrow::csv::ReadOptions,
                        public arrow::csv::ParseOptions,
                        public arrow::csv::ConvertOptions {
 public:
  static CSVConfigHolder *GetCastedHolder(const CSVReadOptions &options) {
    void *holder = options.GetHolder().get();
    return (CSVConfigHolder *) (holder);
  }
};
}  // namespace config
}  // namespace io
}  // namespace cylon

#endif //CYLON_SRC_CYLON_IO_CSV_READ_CONFIG_HOLDER_HPP_
