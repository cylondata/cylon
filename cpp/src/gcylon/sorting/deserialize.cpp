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

#include <glog/logging.h>
#include <gcylon/sorting/deserialize.hpp>
#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <rmm/device_buffer.hpp>
#include <gcylon/cudf_buffer.hpp>
#include <gcylon/net/cudf_net_ops.hpp>

namespace gcylon {

TableDeserializer::TableDeserializer(const cudf::table_view &tv) : tv_(tv) {
}

std::unique_ptr<cudf::table>
TableDeserializer::deserializeTable(std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                    std::vector<int32_t> &disp_per_buffer,
                                    std::vector<int32_t> &buffer_sizes) {

    std::vector<std::unique_ptr<cudf::column>> columns{};
    int32_t num_rows = gcylon::net::numOfRows(tv_.column(0).type(),
                                              buffer_sizes[0],
                                              buffer_sizes[2]);

    for (int i = 0, bc = 0; i < tv_.num_columns(); ++i, bc += 3) {
        // unfortunately this device_buffer performs data copying according to their docs
        // however, there is no other constructor to create a device_buffer using already existing memory on the gpu
        uint8_t * data_buffer = received_buffers[bc]->GetByteBuffer() + disp_per_buffer[bc];
        uint8_t * mask_buffer = received_buffers[bc + 1]->GetByteBuffer() + disp_per_buffer[bc + 1];
        uint8_t * offsets_buffer = received_buffers[bc + 2]->GetByteBuffer() + disp_per_buffer[bc + 2];

        auto db = std::make_shared<rmm::device_buffer>(data_buffer, buffer_sizes[bc], rmm::cuda_stream_default);
        auto mb = std::make_shared<rmm::device_buffer>(mask_buffer, buffer_sizes[bc + 1], rmm::cuda_stream_default);
        auto ob = std::make_shared<rmm::device_buffer>(offsets_buffer, buffer_sizes[bc + 2], rmm::cuda_stream_default);
        rmm::cuda_stream_default.synchronize();

        std::unique_ptr<cudf::column> clmn = gcylon::constructColumn(db,
                                                                     mb,
                                                                     ob,
                                                                     tv_.column(i).type(),
                                                                     num_rows);
        if (clmn == nullptr) {
            std::string msg = "Following column is not constructed successfully: ";
            throw msg + std::to_string(i);
        }
        columns.push_back(std::move(clmn));
    }

    return std::make_unique<cudf::table>(std::move(columns));
}

std::vector<int32_t> displacementsPerTable(std::vector<std::vector<int32_t>> &displacements_per_buffer,
                                           int tableNo) {
    std::vector<int32_t> disp;
    for (long unsigned int i = 0; i < displacements_per_buffer.size(); ++i) {
        disp.push_back(displacements_per_buffer.at(i).at(tableNo));
    }
    return disp;
}

bool allZero(std::vector<int32_t> &buffer_sizes) {
    return std::all_of(buffer_sizes.begin(), buffer_sizes.end(), [](int32_t i){return i == 0;});
}

cylon::Status TableDeserializer::deserialize(std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                                 std::vector<std::vector<int32_t>> &displacements_per_buffer,
                                                 std::vector<std::vector<int32_t>> &buffer_sizes_per_table,
                                                 std::vector<std::unique_ptr<cudf::table>> &received_tables) {

    int number_of_tables = buffer_sizes_per_table.size();

    for (int i = 0; i < number_of_tables; ++i) {
        if (allZero(buffer_sizes_per_table.at(i))) {
            continue;
        }
        std::vector<int32_t> disp = displacementsPerTable(displacements_per_buffer, i);
        std::unique_ptr<cudf::table> outTable = deserializeTable(received_buffers,
                                                                 disp,
                                                                 buffer_sizes_per_table.at(i));
        received_tables.push_back(std::move(outTable));
    }

    return cylon::Status::OK();
}

std::unique_ptr<cudf::column> constructColumn(std::shared_ptr<rmm::device_buffer> data_buffer,
                                              std::shared_ptr<rmm::device_buffer> mask_buffer,
                                              std::shared_ptr<rmm::device_buffer> offsets_buffer,
                                              cudf::data_type dt,
                                              int32_t num_rows) {

    std::unique_ptr<cudf::column> clmn = nullptr;

    // if it is non-string column
    if (dt.id() != cudf::type_id::STRING) {
        if (mask_buffer->size() == 0) {
            clmn = std::make_unique<cudf::column>(dt, num_rows, std::move(*data_buffer));
        } else {
            clmn = std::make_unique<cudf::column>(dt,
                                                  num_rows,
                                                  std::move(*data_buffer),
                                                  std::move(*mask_buffer));
        }
        // if it is a string column
    } else {
        // construct chars child column
        auto cdt = cudf::data_type{cudf::type_id::INT8};
        auto chars_column = std::make_unique<cudf::column>(cdt, data_buffer->size(), std::move(*data_buffer));

        auto odt = cudf::data_type{cudf::type_id::INT32};
        auto offsets_column = std::make_unique<cudf::column>(odt, num_rows + 1, std::move(*offsets_buffer));

        std::vector<std::unique_ptr<cudf::column>> children;
        children.emplace_back(std::move(offsets_column));
        children.emplace_back(std::move(chars_column));

        if (mask_buffer->size() > 0) {
            clmn = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                  num_rows,
                                                  std::move(rmm::device_buffer{0, rmm::cuda_stream_default}),
                                                  std::move(*mask_buffer),
                                                  cudf::UNKNOWN_NULL_COUNT,
                                                  std::move(children));
        } else{
            clmn = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                  num_rows,
                                                  std::move(rmm::device_buffer{0, rmm::cuda_stream_default}),
                                                  std::move(rmm::device_buffer{0, rmm::cuda_stream_default}),
                                                  0,
                                                  std::move(children));
        }
    }
    return clmn;
}

int32_t gcylon::net::numOfRows(const cudf::data_type dt,
                               const int32_t data_size,
                               const int32_t offsets_size) {

    if (cudf::is_fixed_width(dt)) {
        return data_size / cudf::size_of(dt);
    } else if (dt.id() == cudf::type_id::STRING) {
        // there are num_rows + 1 offsets value
        // each offset value is 4 bytes
        return (offsets_size - 4) / 4;
    } else {
        throw "Can not determine number_of_rows on the received table";
    }

}

int32_t gcylon::net::numOfRows(const std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                               const int32_t data_type) {
    if (received_buffers.size() < 3 || received_buffers[0]->GetLength() == 0) {
        return 0;
    }

    cudf::data_type dt(static_cast<cudf::type_id>(data_type));
    int32_t data_size = received_buffers[0]->GetLength();
    int32_t offsets_size = received_buffers[2]->GetLength();
    return numOfRows(dt, data_size, offsets_size);
}


cylon::Status deserializeSingleTable(const std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                     const std::vector<int32_t> &data_types,
                                     std::unique_ptr<cudf::table> &out_table) {
    if(data_types.empty()) {
        return cylon::Status(cylon::Code::ExecutionError, "data_types empty.");
    }

    int32_t num_rows = gcylon::net::numOfRows(received_buffers, data_types[0]);
    int32_t num_cols = data_types.size();
    std::vector<std::unique_ptr<cudf::column>> columns{};
    for (int i = 0, bc = 0; i < num_cols; ++i, bc += 3) {
        cudf::data_type dt(static_cast<cudf::type_id>(data_types[i]));
        std::shared_ptr<CudfBuffer> db = std::dynamic_pointer_cast<CudfBuffer>(received_buffers[bc]);
        std::shared_ptr<CudfBuffer> mb = std::dynamic_pointer_cast<CudfBuffer>(received_buffers[bc + 1]);
        std::shared_ptr<CudfBuffer> ob = std::dynamic_pointer_cast<CudfBuffer>(received_buffers[bc + 2]);

        std::unique_ptr<cudf::column> clmn = constructColumn(db->getBuf(),
                                                             mb->getBuf(),
                                                             ob->getBuf(),
                                                             dt,
                                                             num_rows);
        if (clmn == nullptr) {
            std::string msg = "Following column is not constructed successfully: ";
            throw msg + std::to_string(i);
        }
        columns.push_back(std::move(clmn));
    }

    out_table = std::make_unique<cudf::table>(std::move(columns));

    return cylon::Status::OK();
}


} // end of namespace gcylon