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

namespace gcylon {

/**
 * Deserializer of tables gathered by MPI gather operation
 */
class TableDeserializer {
public:
    /**
     * Provide the local table_view to deconstruct the received tables as a template
     * @param tv
     */
    TableDeserializer(const cudf::table_view &tv);

    /**
     * deserialize a single table
     * @param received_buffers received buffers by gather operation
     * @param disp_per_buffer displacements in buffers for this table
     * @param buffer_sizes buffer sizes for this table
     * @return
     */
    std::unique_ptr<cudf::table> deserializeTable(std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                                  std::vector<int32_t> &disp_per_buffer,
                                                  std::vector<int32_t> &buffer_sizes);

     /**
      * deserialize all tables received by gather operation
      *
      * @param received_buffers received buffers by gather operation
      * @param displacements_per_buffer displacements in all buffer
      * @param buffer_sizes_per_table buffer sizes per table
      * @param received_tables deserialized tables will be saved into this vector
      * @return
      */
    cylon::Status deserialize(std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                              std::vector<std::vector<int32_t>> &displacements_per_buffer,
                              std::vector<std::vector<int32_t>> &buffer_sizes_per_table,
                              std::vector<std::unique_ptr<cudf::table>> &received_tables);

private:
    cudf::table_view tv_;
};

/**
 * Deserialize a column received over the wire
 * @param data_buffer
 * @param mask_buffer
 * @param offsets_buffer
 * @param dt
 * @param num_rows
 * @return
 */
std::unique_ptr<cudf::column> constructColumn(std::shared_ptr<rmm::device_buffer> data_buffer,
                                              std::shared_ptr<rmm::device_buffer> mask_buffer,
                                              std::shared_ptr<rmm::device_buffer> offsets_buffer,
                                              cudf::data_type dt,
                                              int32_t num_rows);

/**
 * deserialize a single table received over the wire
 * @param received_buffers
 * @param data_types
 * @return
 */
cylon::Status deserializeSingleTable(const std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                     const std::vector<int32_t> &data_types,
                                     std::unique_ptr<cudf::table> &out_table);


} // end of namespace gcylon

#endif //GCYLON_DESERIALIZE_HPP
