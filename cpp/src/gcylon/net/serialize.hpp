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

#ifndef GCYLON_CUDF_TABLE_SERIALIZE_HPP
#define GCYLON_CUDF_TABLE_SERIALIZE_HPP

#include <cylon/net/serialize.hpp>
#include <memory>
#include <type_traits>
#include <cudf/table/table_view.hpp>

namespace gcylon {

/**
 * Serialize a CuDF table to send over the wire
 */
class CudfTableSerializer {
public:
    /**
     * @param tv table_view to serialize
     */
    CudfTableSerializer(cudf::table_view &tv);

    /**
     * get the buffer sizes for this table in bytes
     * starting from column 0 to the last column
     * For each column, three buffer sizes are returned in this order:
     *      size of the column data buffer in bytes
     *      size of the column null mask buffer in bytes
     *      size of the column offsets buffer in bytes
     * If there are two columns in a table, 6 data buffers sizes are returned
     *      buffer[0] = size of the data buffer of the first column
     *      buffer[1] = size of the null mask buffer of the first column
     *      buffer[2] = size of the offsets buffer of the first column
     *      buffer[3] = size of the data buffer of the second column
     *      buffer[4] = size of the null mask buffer of the second column
     *      buffer[5] = size of the offsets buffer of the second column
     *
     * If there are n columns, 3 * n buffer sizes are returned
     *
     * This method is symmetrical to getDataBuffers()
     * @return
     */
    std::vector<int32_t> & getBufferSizes();

    /**
     * length of the buffer sizes
     * @return
     */
    int getBufferSizesLength();

    /**
     * zeros for all column data as if the table is empty
     * This is used by the MPI gather root
     * @return
     */
    std::vector<int32_t> getEmptyTableBufferSizes();

    /**
     * Get data buffers starting from column 0 to the last column
     * For each column, three buffers are returned in this order:
     *      column data buffer
     *      column null mask buffer
     *      column offsets buffer
     * If there are two columns in a table, 6 data buffers are returned
     *      buffer[0] = data buffer of the first column
     *      buffer[1] = null mask buffer of the first column
     *      buffer[2] = offsets buffer of the first column
     *      buffer[3] = data buffer of the second column
     *      buffer[4] = null mask buffer of the second column
     *      buffer[5] = offsets buffer of the second column
     *
     * If there are n columns, 3 * n buffers are returned
     *
     * This method is symmetrical to getBufferSizes()
     * @return
     */
    std::vector<uint8_t *> & getDataBuffers();

    /**
     * Get the column data size in bytes and its data buffer
     * @param column_index
     * @return
     */
    std::pair<int32_t, uint8_t *> getColumnData(int column_index);

    /**
     * Get the column null mask size in bytes and its null mask buffer
     * @param column_index
     * @return
     */
    std::pair<int32_t, uint8_t *> getColumnMask(int column_index);

    /**
     * Get the column offsets size in bytes and its offsets buffer
     * @param column_index
     * @return
     */
    std::pair<int32_t, uint8_t *> getColumnOffsets(int column_index);

private:
    cudf::table_view tv_;
    std::vector<int32_t> buffer_sizes_ = std::vector<int32_t>();
    std::vector<uint8_t *> table_buffers_ = std::vector<uint8_t *>();
    bool table_buffers_initialized = false;

    void initTableBuffers();
};


} // end of namespace gcylon

#endif //GCYLON_CUDF_TABLE_SERIALIZE_HPP
