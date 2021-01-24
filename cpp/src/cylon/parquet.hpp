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

#ifndef CYLON_SRC_IO_PARQUET_H_
#define CYLON_SRC_IO_PARQUET_H_

#include <memory>
#include <string>

#include "io/parquet_config.hpp"
#include "table.hpp"
#include "status.hpp"

namespace cylon {
/**
* Create a table by reading a parquet file
* @param path file path
* @return a pointer to the table
*/
Status FromParquet(std::shared_ptr<cylon::CylonContext> &ctx, const std::string &path,
                   std::shared_ptr<Table> &tableOut);
/**
* Read multiple parquet files into multiple tables. If threading is enabled, the tables will be read
* in parallel
* @param ctx
* @param paths
* @param tableOuts
* @param options
* @return
*/
Status FromParquet(std::shared_ptr<cylon::CylonContext> &ctx, const std::vector<std::string> &paths,
                   const std::vector<std::shared_ptr<Table> *> &tableOuts,
                   io::config::ParquetOptions options = cylon::io::config::ParquetOptions());

}  // namespace cylon
#endif //CYLON_SRC_IO_PARQUET_H_
