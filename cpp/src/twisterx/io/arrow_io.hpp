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

#ifndef TWISTERX_SRC_IO_ARROW_IO_H_
#define TWISTERX_SRC_IO_ARROW_IO_H_

#include <string>
#include "csv_read_config.h"
#include "../ctx/twisterx_context.h"
namespace twisterx {
namespace io {

arrow::Result<std::shared_ptr<arrow::Table>> read_csv(twisterx::TwisterXContext *ctx,
                                                      const std::string &path,
                                                      twisterx::io::config::CSVReadOptions options = twisterx::io::config::CSVReadOptions());

}
}

#endif //TWISTERX_SRC_IO_ARROW_IO_H_
