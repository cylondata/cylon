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

#ifndef CYLON_NET_TABLE_SERIALIZE_HPP
#define CYLON_NET_TABLE_SERIALIZE_HPP

#include <vector>

namespace cylon {

/**
 * Serialize a table to send over the wire
 */
class TableSerializer {
public:

  virtual ~TableSerializer() = default;

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
  virtual const std::vector<int32_t>& getBufferSizes() = 0;

  /**
   * length of the buffer sizes
   * @return
   */
  virtual int getNumberOfBuffers() = 0;

  /**
   * zeros for all column data as if the table is empty
   * This is used by the MPI gather root
   * @return
   */
  virtual std::vector<int32_t> getEmptyTableBufferSizes() = 0;

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
  virtual const std::vector<const uint8_t *>& getDataBuffers() = 0;

  /**
   * Return data types of the columns in an int vector
   * This vector will be sent over the over
   * Data type of the first column will be at the first position of the vector,
   * Data type of the second column will be at the second position of the vector,
   * so on,
   * @return
   */
  virtual std::vector<int32_t> getDataTypes() = 0;

};

} // end of namespace cylon

#endif //CYLON_NET_TABLE_SERIALIZE_HPP
