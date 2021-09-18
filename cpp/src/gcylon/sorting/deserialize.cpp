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

namespace gcylon {

TableDeserializer::TableDeserializer(cudf::table_view &tv) : tv_(tv) {
}

std::unique_ptr<cudf::column> TableDeserializer::constructColumn(uint8_t * data_buffer,
                                                                 int32_t data_size,
                                                                 uint8_t * mask_buffer,
                                                                 int32_t mask_size,
                                                                 uint8_t * offsets_buffer,
                                                                 int32_t offsets_size,
                                                                 cudf::data_type dt) {

    std::unique_ptr<cudf::column> clmn = nullptr;

    // unfortunately this device_buffer performs data copying according to their docs
    // however, there is no other constructor to create a device_buffer using already existing memory on the gpu
    rmm::device_buffer db(data_buffer, data_size, rmm::cuda_stream_default);

    // if it is non-string column
    if (dt.id() != cudf::type_id::STRING) {
        int32_t data_count = data_size / cudf::size_of(dt);
        if (mask_size == 0) {
            rmm::cuda_stream_default.synchronize();
            clmn = std::make_unique<cudf::column>(dt, data_count, std::move(db));
        } else {
            rmm::device_buffer mb(mask_buffer, mask_size, rmm::cuda_stream_default);
            rmm::cuda_stream_default.synchronize();
            clmn = std::make_unique<cudf::column>(dt,
                                                  data_count,
                                                  std::move(db),
                                                  std::move(mb));
        }
    // if it is a string column
    } else {
        rmm::device_buffer ob(offsets_buffer, offsets_size, rmm::cuda_stream_default);

        // construct chars child column
        auto cdt = cudf::data_type{cudf::type_id::INT8};
        auto chars_column = std::make_unique<cudf::column>(cdt, data_size, std::move(db));

        auto odt = cudf::data_type{cudf::type_id::INT32};
        int offsets_count = offsets_size / 4;
        auto offsets_column = std::make_unique<cudf::column>(odt, offsets_count, std::move(ob));

        std::vector<std::unique_ptr<cudf::column>> children;
        children.emplace_back(std::move(offsets_column));
        children.emplace_back(std::move(chars_column));

        if (mask_size > 0) {
            rmm::device_buffer rmm_buf{0, rmm::cuda_stream_default, rmm::mr::get_current_device_resource()};
            rmm::device_buffer mb(mask_buffer, mask_size, rmm::cuda_stream_default);
            rmm::cuda_stream_default.synchronize();
            clmn = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                    offsets_count - 1,
                                                    std::move(rmm_buf),
                                                    std::move(mb),
                                                    cudf::UNKNOWN_NULL_COUNT,
                                                    std::move(children));
        } else{
            rmm::cuda_stream_default.synchronize();
            clmn = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                    offsets_count -1,
                                                    std::move(rmm::device_buffer{0, rmm::cuda_stream_default}),
                                                    std::move(rmm::device_buffer{0, rmm::cuda_stream_default}),
                                                    0,
                                                    std::move(children));
        }
    }
    return clmn;
}

std::unique_ptr<cudf::table>
TableDeserializer::deserializeTable(std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                                    std::vector<int32_t> &disp_per_buffer,
                                    std::vector<int32_t> &buffer_sizes) {

    std::vector<std::unique_ptr<cudf::column>> columns{};
    int bc = 0;
    for (int i = 0; i < tv_.num_columns(); ++i) {
        std::unique_ptr<cudf::column> clmn =
                constructColumn(received_buffers[bc]->GetByteBuffer() + disp_per_buffer[bc],
                                buffer_sizes[bc],
                                received_buffers[bc + 1]->GetByteBuffer() + disp_per_buffer[bc + 1],
                                buffer_sizes[bc + 1],
                                received_buffers[bc + 2]->GetByteBuffer() + disp_per_buffer[bc + 2],
                                buffer_sizes[bc + 2],
                                tv_.column(i).type());
        bc += 3;
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
    for (int i: buffer_sizes) {
        if (i != 0) {
            return false;
        }
    }
    return true;
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

} // end of namespace gcylon