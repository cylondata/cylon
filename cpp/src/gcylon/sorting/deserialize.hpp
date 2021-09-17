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

#ifndef GCYLON_DESERIALIZE_HPP
#define GCYLON_DESERIALIZE_HPP

#include <memory>
#include <cudf/table/table_view.hpp>
#include <cylon/net/buffer.hpp>
#include <rmm/device_buffer.hpp>

namespace gcylon {


class TableDeserializer {
public:
    TableDeserializer(cudf::table_view &tv);

    std::unique_ptr<cudf::column> constructColumn(uint8_t *data_buffer,
                                                  int32_t data_size,
                                                  uint8_t *mask_buffer,
                                                  int32_t mask_size,
                                                  uint8_t *offsets_buffer,
                                                  int32_t offsets_size,
                                                  cudf::data_type dt);

    std::unique_ptr<cudf::table>  deserializeTable(std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                   std::vector<int32_t> &disp_per_buffer,
                                   std::vector<int32_t> &buffer_sizes);

    /**
     * deserialize all tables received by gather operation
     */
    cylon::Status deserialize(std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                              std::vector<std::vector<int32_t>> &displacements_per_buffer,
                              std::vector<std::vector<int32_t>> &buffer_sizes_per_table,
                              std::vector<std::unique_ptr<cudf::table>> &received_tables);

private:
    cudf::table_view tv_;
};


} // end of namespace gcylon

#endif //GCYLON_DESERIALIZE_HPP
