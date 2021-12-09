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

#ifndef CYLON_CPP_SRC_CYLON_NET_TABLE_SERIALIZE_HPP_
#define CYLON_CPP_SRC_CYLON_NET_TABLE_SERIALIZE_HPP_

#include "cylon/table.hpp"
#include "cylon/net/serialize.hpp"

namespace cylon{

Status MakeTableSerializer(const std::shared_ptr<Table>& table,
                           std::shared_ptr<TableSerializer>* serializer);

Status DeserializeTable(const std::shared_ptr<CylonContext> &ctx,
                        const std::shared_ptr<arrow::Schema> &schema,
                        const std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                        const std::vector<int32_t> &disp_per_buffer,
                        const std::vector<int32_t> &buffer_sizes,
                        std::shared_ptr<Table> *output);
}

#endif //CYLON_CPP_SRC_CYLON_NET_TABLE_SERIALIZE_HPP_
