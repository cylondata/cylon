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

#ifndef CYLON_SRC_IO_ARROW_IO_H_
#define CYLON_SRC_IO_ARROW_IO_H_

#include <arrow/api.h>
#include <string>

#include "csv_read_config.hpp"
#include "../ctx/cylon_context.hpp"

namespace cylon {
namespace io {

arrow::Result<std::shared_ptr<arrow::Table>> read_csv(cylon::CylonContext *ctx,
                   const std::string &path,
                   cylon::io::config::CSVReadOptions options = cylon::io::config::CSVReadOptions());

}  // namespace io
}  // namespace cylon

#endif //CYLON_SRC_IO_ARROW_IO_H_
