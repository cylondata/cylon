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
#include <gcylon/net/cudf_serialize.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/null_mask.hpp>

namespace gcylon {

CudfTableSerializer::CudfTableSerializer(cudf::table_view &tv) : tv_(tv) {
}

void CudfTableSerializer::initTableBuffers() {
    // for each column, we keep 3 data objects: data, mask, offsets
    std::pair<int32_t, uint8_t *> p;
    for (int i = 0; i < tv_.num_columns(); ++i) {
        p = getColumnData(i);
        buffer_sizes_.push_back(p.first);
        table_buffers_.push_back(p.second);

        p = getColumnMask(i);
        buffer_sizes_.push_back(p.first);
        table_buffers_.push_back(p.second);

        p = getColumnOffsets(i);
        buffer_sizes_.push_back(p.first);
        table_buffers_.push_back(p.second);
    }
    table_buffers_initialized = true;
}

std::vector<int32_t> CudfTableSerializer::getBufferSizes() {
    if(!table_buffers_initialized) {
        initTableBuffers();
    }

    return buffer_sizes_;
}

std::vector<uint8_t *> CudfTableSerializer::getDataBuffers() {
    if(!table_buffers_initialized) {
        initTableBuffers();
    }
    return table_buffers_;
}

int CudfTableSerializer::getNumberOfBuffers() {
    return tv_.num_columns() * 3;
}

std::vector<int32_t> CudfTableSerializer::getEmptyTableBufferSizes() {
    return std::vector<int32_t>(getNumberOfBuffers(), 0);
}

std::pair<int32_t, uint8_t *> CudfTableSerializer::getColumnData(int column_index) {
    auto cv = tv_.column(column_index);
    if (cv.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view scv(cv);
        return std::make_pair(scv.chars_size(), (uint8_t *)scv.chars().data<uint8_t>());
    }

    int size = cudf::size_of(cv.type()) * cv.size();
    return std::make_pair(size, (uint8_t *)cv.data<uint8_t>());
}

std::pair<int32_t, uint8_t *> CudfTableSerializer::getColumnMask(int column_index) {
    auto cv = tv_.column(column_index);
    if (cv.has_nulls()) {
        int size = cudf::bitmask_allocation_size_bytes(cv.size());
        return std::make_pair(size, (uint8_t *)cv.null_mask());
    } else {
        return std::make_pair(0, nullptr);
    }
}

std::pair<int32_t, uint8_t *> CudfTableSerializer::getColumnOffsets(int column_index) {
    auto cv = tv_.column(column_index);
    if (cv.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view scv(cv);
        int size = cudf::size_of(scv.offsets().type()) * scv.offsets().size();
        return std::make_pair(size, (uint8_t *)scv.offsets().data<uint8_t>());
    }

    return std::make_pair(0, nullptr);
}

} // end of namespace gcylon