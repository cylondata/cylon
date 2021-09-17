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

#ifndef GCYLON_SERIALIZE_HPP
#define GCYLON_SERIALIZE_HPP

#include <memory>
#include <type_traits>
#include <cudf/table/table_view.hpp>

namespace gcylon {


template <typename E>
constexpr auto to_underlying(E e) noexcept
{
    return static_cast<std::underlying_type_t<E>>(e);
}

/**
 * enums to encode the column data.
 * enum values indicate indices in the encoded array after the starting point
 */
enum class ColumnHeaderEncoding : int32_t {
    DATA_SIZE,  /// in the number of bytes
    MASK_SIZE, /// in the number of bytes
    OFFSETS_SIZE, /// in the number of bytes
    COUNT, /// this is to show the number of enums, it has to the last in the list
};


class TableSerializer {
public:
    TableSerializer(cudf::table_view &tv);
    std::vector<int32_t> & getBufferSizes();
    int getBufferSizesLength();
    std::vector<int32_t> getEmptyTableBufferSizes();
    std::vector<uint8_t *> & getDataBuffers();

    std::pair<int32_t, uint8_t *> getColumnData(int column_index);
    std::pair<int32_t, uint8_t *> getColumnMask(int column_index);
    std::pair<int32_t, uint8_t *> getColumnOffsets(int column_index);

private:
    cudf::table_view tv_;
    std::vector<int32_t> buffer_sizes_ = std::vector<int32_t>();
    std::vector<uint8_t *> table_buffers_ = std::vector<uint8_t *>();
    bool table_buffers_initialized = false;

    void initTableBuffers();
};


} // end of namespace gcylon

#endif //GCYLON_SERIALIZE_HPP
