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
#ifndef GCYLON_CUDF_ALL_TO_ALL_H
#define GCYLON_CUDF_ALL_TO_ALL_H

#include <unordered_map>

#include <cylon/net/ops/all_to_all.hpp>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/status.hpp>
#include <cylon/net/buffer.hpp>

#include <gcylon/cudf_buffer.hpp>
#include <cudf/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace gcylon {

cudf::size_type dataLength(cudf::column_view const& cw);

class PendingBuffer {
public:
  PendingBuffer(const uint8_t *buffer,
                int buffer_size,
                int target,
                std::unique_ptr<int []> headers = nullptr,
                int headers_length = -1);

  PendingBuffer(int target,
                std::unique_ptr<int []> headers,
                int headers_length);

  bool sendBuffer(std::shared_ptr<cylon::AllToAll> all);

private:
  const uint8_t *buffer;
  int buffer_size;
  int target;
  std::unique_ptr<int []> headers;
  int headers_length;
};

/**
 * column view for CuDF column to get the buffer to send
 */
class PartColumnView {
public:
  PartColumnView(const cudf::column_view &cv, const std::vector<cudf::size_type> &part_indexes);

  const uint8_t * getDataBuffer(int part_index);
  int getDataBufferSize(int part_index);

  const uint8_t * getOffsetBuffer(int part_index);
  int getOffsetBufferSize(int part_index);

  const uint8_t * getMaskBuffer(int part_index);
  int getMaskBufferSize(int part_index);

  inline int numberOfElements(int part_index) {
      return part_indexes[part_index + 1] - part_indexes[part_index];
  }

  inline int getColumnTypeId() {
    return (int)cv.type().id();
  }

  inline cudf::data_type getColumnDataType() {
    return cv.type();
  }

  inline const cudf::column_view & getColumnView() {
      return cv;
  }

private:
  // private members
  const cudf::column_view &cv;
  std::unique_ptr<cudf::strings_column_view> scv;
  // partition indices, last one shows the limit of the previous one
  // there are part_indexes.size()-1 partitions
  const std::vector<cudf::size_type> & part_indexes;

  // partition char offsets on a single char array for the column
  // last index shows the end of the last string
  std::vector<cudf::size_type> part_char_offsets;

  // this is to prevent std::shared_ptr<rmm::device_buffer> to be deleted before they are sent out
  std::unordered_map<long unsigned int, rmm::device_buffer> mask_buffers{};
};

/**
 * partitioned cudf table view
 * to get partitioned column buffers to be sent out
 */
class PartTableView {
public:
  PartTableView(const cudf::table_view &tv, const std::vector<cudf::size_type> &part_indexes);

  std::shared_ptr<PartColumnView> column(int column_index);

  inline int numberOfColumns() {
    return tv.num_columns();
  }

  inline int numberOfRows(int part_index) {
    return part_indexes[part_index + 1] - part_indexes[part_index];
  }

  inline int numberOfParts() {
    return part_indexes.size() - 1;
  }

private:
  // private members
  cudf::table_view tv;

  // partitioned column views
  std::unordered_map<int, std::shared_ptr<PartColumnView>> columns{};

  // partition indices, last one shows the limit of the previous one
  const std::vector<cudf::size_type> part_indexes;
};

struct PendingReceives {
  // table variables
  // currently received columns
  std::vector<std::unique_ptr<cudf::column>> columns;
  // number of columns in the table
  int number_of_columns{-1};
  // the reference
  int reference{-1};

  // column variables
  // data type of the column
  int has_data_buff{false};
  // the current data column index
  int column_index{-1};
  // whether the current column has the null buffer
  bool has_null_buffer{false};
  // whether the current column has the offset buffer
  bool has_offset_buffer{false};
  // number of data elements
  int data_size{0};
  // length of the data buffer
  int data_buffer_len{0};


  // data buffer for the current column
  std::shared_ptr<rmm::device_buffer> data_buffer;
  // null buffer for the current column
  std::shared_ptr<rmm::device_buffer> null_buffer;
  // offsets buffer for the current column
  std::shared_ptr<rmm::device_buffer> offsets_buffer;
};

/**
 * This function is called when a table is fully received
 * @param source the source
 * @param table the table that is received
 * @param reference the reference number sent by the sender
 * @return true if we accept this buffer
 */
using CudfCallback = std::function<bool(int source, std::unique_ptr<cudf::table> table, int reference)>;

class CudfAllToAll : public cylon::ReceiveCallback {

public:
  CudfAllToAll(std::shared_ptr<cylon::CylonContext> ctx,
               const std::vector<int> &sources,
               const std::vector<int> &targets,
               int edge_id,
               CudfCallback callback);

  /**
   * Insert a table to be sent, if the table is accepted return true
   *
   * @param tview the table to send
   * @param target the target to send the table
   * @return true if the buffer is accepted
   */
  int insert(const std::shared_ptr<cudf::table_view> &tview, int32_t target);

  /**
   * Insert a table to be sent, if the table is accepted return true
   *
   * @param tview the table to send
   * @param target the target to send the table
   * @param reference a reference that can be sent in the header
   * @return true if the buffer is accepted
   */
  int insert(const std::shared_ptr<cudf::table_view> &tview, int32_t target, int32_t reference);

  /**
   * Insert a partitioned table to be sent, if the table is accepted return true
   * Can not sent more than one partitioned table at once,
   * before finishing to send a partitioned table, if another one is asked,
   * zero is returned
   *
   * @param tview partitioned table to be sent
   * @param offsets partitioning offsets in the table
   * @return true if the buffer is accepted
   */
  int insert(const cudf::table_view &tview, const std::vector<cudf::size_type> &offsets, int ref);

  /**
   * Check weather the operation is complete, this method needs to be called until the operation is complete
   * @return true if the operation is complete
   */
  bool isComplete();

  /**
   * When this function is called, the operation finishes at both receivers and targets
   * @return
   */
  void finish();

  /**
   * Close the operation
   */
  void close();

  /**
   * This function is called when a data is received
   */
  bool onReceive(int source, std::shared_ptr<cylon::Buffer> buffer, int length) override;

  /**
   * Receive the header, this happens before we receive the actual data
   */
  bool onReceiveHeader(int source, int finished, int *buffer, int length) override;

  /**
   * This method is called after we successfully send a buffer
   * @return
   */
  bool onSendComplete(int target, const void *buffer, int length) override;

private:
  std::unique_ptr<int []> makeTableHeader(int headers_length, int ref, int number_of_columns, int number_of_rows);

  void makePartTableBuffers(int target,
                            int ref,
                            std::queue<std::shared_ptr<PendingBuffer>> &buffer_queue);

  void makeTableBuffers(std::shared_ptr<cudf::table_view> tview,
                        int target,
                        int ref,
                        std::queue<std::shared_ptr<PendingBuffer>> &buffer_queue);

  std::unique_ptr<int []> makeColumnHeader(int headers_length,
                                           int column_index,
                                           bool has_data_buffer,
                                           bool has_mask,
                                           bool has_offset,
                                           int number_of_elements);

  void makePartColumnBuffers(std::shared_ptr<PartColumnView> pcv,
                         int part_index,
                         int column_index,
                         int target,
                         std::queue<std::shared_ptr<PendingBuffer>> &buffer_queue);

  void makeColumnBuffers(const cudf::column_view &cw,
                         int column_index,
                         int target,
                         std::queue<std::shared_ptr<PendingBuffer>> &buffer_queue);

  void constructColumn(std::shared_ptr<PendingReceives> pr);

  std::unique_ptr<cudf::table> constructTable(std::shared_ptr<PendingReceives> pr);

  /**
   * worker rank
   */
  int rank_;

  /**
   * The sources
   */
  std::vector<int> sources_;

  /**
   * The targets
   */
  std::vector<int> targets_;

  /**
   * The underlying alltoall communication
   */
  std::shared_ptr<cylon::AllToAll> all_;

  /**
   * we keep a queue for each target
   */
  std::unordered_map<int, std::queue<std::shared_ptr<PendingBuffer>>> send_queues_{};

  /**
   * Keep track of the receives
   */
  std::unordered_map<int, std::shared_ptr<PendingReceives>> receives_{};

  /**
   * this is the allocator to create memory when receiving
   */
  CudfAllocator * allocator_;

  /**
   * inform the callback when a table received
   */
  CudfCallback recv_callback_;

  /**
   * We have received the finish
   */
  bool finished_ = false;

  bool completed_ = false;
  bool finish_called_ = false;

  std::unique_ptr<PartTableView> ptview_;
};

}// end of namespace gcylon

#endif //GCYLON_CUDF_ALL_TO_ALL_H
