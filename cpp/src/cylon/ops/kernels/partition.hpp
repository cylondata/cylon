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

#ifndef CYLON_SRC_CYLON_OPS_KERNELS_PARTITION_HPP_
#define CYLON_SRC_CYLON_OPS_KERNELS_PARTITION_HPP_

#include <status.hpp>
#include <vector>
#include <unordered_map>
#include <memory>
#include <table.hpp>
namespace cylon {
namespace kernel {
//Status HashPartition(std::shared_ptr<CylonContext> &ctx, const std::shared_ptr<Table> &table,
//                     const std::vector<int> &hash_columns, int no_of_partitions,
//                     std::vector<std::shared_ptr<Table>> &out);
//
//Status HashPartition(std::shared_ptr<CylonContext> &ctx, const std::shared_ptr<Table> &table,
//                     int hash_column, int no_of_partitions,
//                     std::unordered_map<int, std::shared_ptr<cylon::Table>> *out);
}
}

#endif //CYLON_SRC_CYLON_OPS_KERNELS_PARTITION_HPP_
