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

#include <cudf/types.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace gcylon {

cudf::size_type dataLength(cudf::column_view const& cw);

class CudfBuffer : public cylon::Buffer {
public:
    CudfBuffer(std::shared_ptr<rmm::device_buffer> rmmBuf);
    int64_t GetLength() override;
    uint8_t * GetByteBuffer() override;
    std::shared_ptr<rmm::device_buffer> getBuf() const;
private:
    std::shared_ptr<rmm::device_buffer> rmmBuf;
};

class CudfAllocator : public cylon::Allocator {
public:
    cylon::Status Allocate(int64_t length, std::shared_ptr<cylon::Buffer> *buffer) override;
    virtual ~CudfAllocator();
};

class PendingBuffer {
public:
    PendingBuffer(const uint8_t *buffer,
                  int bufferSize,
                  int target,
                  std::unique_ptr<int []> headers = nullptr,
                  int headersLength = -1);

    PendingBuffer(int target,
                  std::unique_ptr<int []> headers,
                  int headersLength);

    bool sendBuffer(std::shared_ptr<cylon::AllToAll> all);

private:
    const uint8_t *buffer;
    int bufferSize;
    int target;
    std::unique_ptr<int []> headers;
    int headersLength;
};

/**
 * column view for CuDF column to get the buffer to send
 */
class PartColumnView {
public:
    PartColumnView(const cudf::column_view &cv, const std::vector<cudf::size_type> &partIndexes);

    const uint8_t * getDataBuffer(int partIndex);
    int getDataBufferSize(int partIndex);

    const uint8_t * getOffsetBuffer(int partIndex);
    int getOffsetBufferSize(int partIndex);

    const uint8_t * getMaskBuffer(int partIndex);
    int getMaskBufferSize(int partIndex);

    inline int numberOfElements(int partIndex) {
        return partIndexes[partIndex + 1] - partIndexes[partIndex];
    }

    inline int getColumnTypeId() {
        return (int)cv.type().id();
    }

    inline const cudf::column_view & getColumnView() {
        return cv;
    }

private:
    // private members
    const cudf::column_view &cv;
    std::unique_ptr<cudf::strings_column_view> scv;
    // partition indices, last one shows the limit of the previous one
    // there are partIndexes.size()-1 partitions
    const std::vector<cudf::size_type> & partIndexes;

    // partition char offsets on a single char array for the column
    // last index shows the end of the last string
    std::vector<cudf::size_type> partCharOffsets;

    // this is to prevent std::shared_ptr<rmm::device_buffer> to be deleted before they are sent out
    std::unordered_map<long unsigned int, rmm::device_buffer> maskBuffers{};
};

/**
 * partitioned cudf table view
 * to get partitioned column buffers to be sent out
 */
class PartTableView {
public:
    PartTableView(cudf::table_view &tv, std::vector<cudf::size_type> &partIndexes);

    std::shared_ptr<PartColumnView> column(int columnIndex);

    inline int numberOfColumns() {
        return tv.num_columns();
    }

    inline int numberOfRows(int partIndex) {
        return partIndexes[partIndex + 1] - partIndexes[partIndex];
    }

    inline int numberOfParts() {
        return partIndexes.size() - 1;
    }

private:
    // private members
    cudf::table_view tv;

    // partitioned column views
    std::unordered_map<int, std::shared_ptr<PartColumnView>> columns{};

    // partition indices, last one shows the limit of the previous one
    // there are partIndexes.size()-1 partitions
    std::vector<cudf::size_type> & partIndexes;
};

struct PendingReceives {
    // table variables
    // currently received columns
    std::unordered_map<int, std::unique_ptr<cudf::column>> columns;
    // number of columns in the table
    int numberOfColumns{-1};
    // the reference
    int reference{-1};

    // column variables
    // data type of the column
    int columnDataType{-1};
    // the current data column index
    int columnIndex{-1};
    // whether the current column has the null buffer
    bool hasNullBuffer{false};
    // whether the current column has the offset buffer
    bool hasOffsetBuffer{false};
    // number of data elements
    int dataSize{0};
    // length of the data buffer
    int dataBufferLen{0};


    // data buffer for the current column
    std::shared_ptr<rmm::device_buffer> dataBuffer;
    // null buffer for the current column
    std::shared_ptr<rmm::device_buffer> nullBuffer;
    // offsets buffer for the current column
    std::shared_ptr<rmm::device_buffer> offsetsBuffer;
};

/**
 * This function is called when a table is fully received
 * @param source the source
 * @param table the table that is received
 * @param reference the reference number sent by the sender
 * @return true if we accept this buffer
 */
using CudfCallback = std::function<bool(int source, const std::shared_ptr<cudf::table> &table, int reference)>;

class CudfAllToAll : public cylon::ReceiveCallback {

public:
  CudfAllToAll(std::shared_ptr<cylon::CylonContext> ctx,
               const std::vector<int> &sources,
               const std::vector<int> &targets,
               int edgeId,
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
  int insert(cudf::table_view &tview, std::vector<cudf::size_type> &offsets, int ref);

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
    std::unique_ptr<int []> makeTableHeader(int headersLength, int ref, int numberOfColumns, int numberOfRows);

    void makePartTableBuffers(int target,
                              int ref,
                              std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue);

    void makeTableBuffers(std::shared_ptr<cudf::table_view> tview,
                          int target,
                          int ref,
                          std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue);

    std::unique_ptr<int []> makeColumnHeader(int headersLength,
                                             int columnIndex,
                                             int typeId,
                                             bool hasMask,
                                             bool hasOffset,
                                             int numberOfElements);

    void makePartColumnBuffers(std::shared_ptr<PartColumnView> pcv,
                           int partIndex,
                           int columnIndex,
                           int target,
                           std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue);

    void makeColumnBuffers(const cudf::column_view &cw,
                           int columnIndex,
                           int target,
                           std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue);

    void constructColumn(std::shared_ptr<PendingReceives> pr);

    std::shared_ptr<cudf::table> constructTable(std::shared_ptr<PendingReceives> pr);

    /**
     * worker rank
     */
    int myrank;

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
    std::unordered_map<int, std::queue<std::shared_ptr<PendingBuffer>>> sendQueues{};

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
    bool finished = false;

    bool completed_ = false;
    bool finishCalled_ = false;

    std::unique_ptr<PartTableView> ptview;
};

}// end of namespace gcylon

#endif //GCYLON_CUDF_ALL_TO_ALL_H
