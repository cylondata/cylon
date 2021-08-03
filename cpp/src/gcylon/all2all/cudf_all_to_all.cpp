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
#include <cudf/table/table.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>

#include <gcylon/all2all/cudf_all_to_all.hpp>
#include <gcylon/gtable.hpp>
#include <gcylon/utils/util.hpp>
#include <gcylon/all2all/cudf_all_to_all.cuh>
#include <cylon/net/mpi/mpi_communicator.hpp>

namespace gcylon {

//////////////////////////////////////////////////////////////////////
// global types and fuctions
//////////////////////////////////////////////////////////////////////
// sizes of cudf data types
// ref: type_id in cudf/types.hpp
int type_bytes[] = {0, 1, 2, 4, 8, 1, 2, 4, 8, 4, 8, 1, 4, 8, 8, 8, 8, 4, 8, 8, 8, 8, -1, -1, -1, 4, 8, -1, -1};

/**
 * whether the data type is uniform in size such as int (4 bytes) or string(variable length)
 * @param input
 * @return
 */
int data_type_size(cudf::column_view const& cw){
    return type_bytes[static_cast<int>(cw.type().id())];
}

/**
 * whether the data type is uniform in size such as int (4 bytes) or string(variable length)
 * @param input
 * @return
 */
bool uniform_size_data(cudf::column_view const& cw){
    int dataTypeSize = data_type_size(cw);
    return dataTypeSize == -1 ? false : true;
}

/**
 * data buffer length of a column in bytes
 * @param input
 * @return
 */
cudf::size_type dataLength(cudf::column_view const& cw){
    int elementSize = type_bytes[static_cast<int>(cw.type().id())];
    if (elementSize == -1) {
        std::cout << "ERRORRRRRR unsupported type id: " << static_cast<int>(cw.type().id()) << std::endl;
        return -1;
    }

    // even null values exist in the buffer with unspecified values
    return elementSize * cw.size();
}

//////////////////////////////////////////////////////////////////////
// CudfBuffer implementations
//////////////////////////////////////////////////////////////////////
CudfBuffer::CudfBuffer(std::shared_ptr<rmm::device_buffer> rmmBuf) : rmmBuf(rmmBuf) {}

int64_t CudfBuffer::GetLength() {
  return rmmBuf->size();
}

uint8_t * CudfBuffer::GetByteBuffer() {
  return (uint8_t *)rmmBuf->data();
}

std::shared_ptr<rmm::device_buffer> CudfBuffer::getBuf() const {
  return rmmBuf;
}

//////////////////////////////////////////////////////////////////////
// CudfAllocator implementations
//////////////////////////////////////////////////////////////////////
cylon::Status CudfAllocator::Allocate(int64_t length, std::shared_ptr<cylon::Buffer> *buffer) {
  try {
    auto rmmBuf = std::make_shared<rmm::device_buffer>(length, rmm::cuda_stream_default);
    *buffer = std::make_shared<CudfBuffer>(rmmBuf);
    return cylon::Status::OK();
  } catch (rmm::bad_alloc * badAlloc) {
    LOG(ERROR) << "failed to allocate gpu memory with rmm: " << badAlloc->what();
    return cylon::Status(cylon::Code::GpuMemoryError);
  }
}

CudfAllocator::~CudfAllocator() = default;

//////////////////////////////////////////////////////////////////////
// PendingBuffer implementations
//////////////////////////////////////////////////////////////////////
PendingBuffer::PendingBuffer(const uint8_t *buffer,
                             int bufferSize,
                             int target,
                             std::unique_ptr<int []> headers,
                             int headersLength):
        buffer(buffer),
        bufferSize(bufferSize),
        target(target),
        headers(std::move(headers)),
        headersLength(headersLength) {}

PendingBuffer::PendingBuffer(int target,
                             std::unique_ptr<int []> headers,
                             int headersLength):
        buffer(nullptr),
        bufferSize(-1),
        target(target),
        headers(std::move(headers)),
        headersLength(headersLength) {}

bool PendingBuffer::sendBuffer(std::shared_ptr<cylon::AllToAll> all) {
    // if there is no data buffer, only header buffer
    if (bufferSize < 0) {
        bool accepted = all->insert(nullptr, 0, target, headers.get(), headersLength);
        if (!accepted) {
            LOG(WARNING) << " header buffer not accepted to be sent";
        }
        return accepted;
    }

    // if there is no header buffer, only data buffer
    if (headersLength < 0) {
        bool accepted = all->insert(buffer, bufferSize, target);
        if (!accepted) {
            LOG(WARNING) << " data buffer not accepted to be sent";
        }
        return accepted;
    }

    bool accepted = all->insert(buffer, bufferSize, target, headers.get(), headersLength);
    if (!accepted) {
        LOG(WARNING) << " data buffer with header not accepted to be sent";
    }
    return accepted;
}

//////////////////////////////////////////////////////////////////////
// PartColumnView implementations
//////////////////////////////////////////////////////////////////////
PartColumnView::PartColumnView(const cudf::column_view &cv, const std::vector<cudf::size_type> &partIndexes)
    : cv(cv), partIndexes(partIndexes), partCharOffsets(partIndexes.size()) {

    if (cv.type().id() == cudf::type_id::STRING) {
        scv = std::make_unique<cudf::strings_column_view>(this->cv);

        // get offsets from gpu to cpu concurrently
        int offsetDataTypeSize = cudf::size_of(scv->offsets().type());
        uint8_t * dest = (uint8_t *)partCharOffsets.data();
        const uint8_t * src = scv->offsets().data<uint8_t>();
        for (long unsigned int i = 0; i < partIndexes.size(); ++i) {
            cudaMemcpyAsync(dest + offsetDataTypeSize * i,
                            src + offsetDataTypeSize * partIndexes[i],
                            offsetDataTypeSize,
                            cudaMemcpyDeviceToHost);
        }
        // synch on the default stream
        cudaStreamSynchronize(0);
    }

    if (cv.nullable()) {
        for (long unsigned int i = 0; i < partIndexes.size() -1; ++i) {
            auto maskBuf = cudf::copy_bitmask(cv.null_mask(), partIndexes[i], partIndexes[i+1]);
            maskBuffers.emplace(std::make_pair(i, std::move(maskBuf)));
        }
        rmm::cuda_stream_default.synchronize();
    }
}

const uint8_t * PartColumnView::getDataBuffer(int partIndex) {
    if (cv.type().id() == cudf::type_id::STRING) {
        return scv->chars().data<uint8_t>() + partCharOffsets[partIndex];
    }

    int startPos = cudf::size_of(cv.type()) * partIndexes[partIndex];
    return cv.data<uint8_t>() + startPos;
}

int PartColumnView::getDataBufferSize(int partIndex) {
    if (cv.type().id() == cudf::type_id::STRING) {
        return partCharOffsets[partIndex + 1] - partCharOffsets[partIndex];
    }

    return cudf::size_of(cv.type()) * numberOfElements(partIndex);
}

const uint8_t * PartColumnView::getOffsetBuffer(int partIndex) {
    if (cv.type().id() == cudf::type_id::STRING) {
        return scv->offsets().data<uint8_t>() + partIndexes[partIndex] * cudf::size_of(scv->offsets().type());
    }

    return nullptr;
}

int PartColumnView::getOffsetBufferSize(int partIndex) {
    if (cv.type().id() == cudf::type_id::STRING) {
        if (numberOfElements(partIndex) == 0) {
            return 0;
        } else {
            return (numberOfElements(partIndex) + 1) * cudf::size_of(scv->offsets().type());
        }
    }

    return 0;
}

const uint8_t * PartColumnView::getMaskBuffer(int partIndex) {
    if (!cv.nullable()) {
        return nullptr;
    }
    return (uint8_t *)maskBuffers.at(partIndex).data();
}

int PartColumnView::getMaskBufferSize(int partIndex) {
    if (!cv.nullable())
        return 0;
    return maskBuffers.at(partIndex).size();
}

//////////////////////////////////////////////////////////////////////
// PartTableView implementations
//////////////////////////////////////////////////////////////////////
PartTableView::PartTableView(cudf::table_view &tv, std::vector<cudf::size_type> &partIndexes)
        : tv(tv), partIndexes(partIndexes) {

    // add the limit of the last partition
    partIndexes.push_back(tv.num_rows());

    for (int i = 0; i < this->tv.num_columns(); ++i) {
        auto pcv = std::make_shared<PartColumnView>(this->tv.column(i), this->partIndexes);
        columns.insert(std::make_pair(i, pcv));
    }
}

std::shared_ptr<PartColumnView> PartTableView::column(int columnIndex) {
    return columns.at(columnIndex);
}

//////////////////////////////////////////////////////////////////////
// CudfAllToAll implementations
//////////////////////////////////////////////////////////////////////
CudfAllToAll::CudfAllToAll(std::shared_ptr<cylon::CylonContext> ctx,
                           const std::vector<int> &sources,
                           const std::vector<int> &targets,
                           int edgeId,
                           CudfCallback callback) :
        myrank(ctx->GetRank()),
        sources_(sources),
        targets_(targets),
        recv_callback_(std::move(callback)){

    allocator_ = new CudfAllocator();

    // we need to pass the correct arguments
    all_ = std::make_shared<cylon::AllToAll>(ctx, sources_, targets_, edgeId, this, allocator_);

    // add the trackers for sending
    for (auto t : targets_) {
        sendQueues.insert(std::make_pair(t, std::queue<std::shared_ptr<PendingBuffer>>()));
    }

    for (auto t : sources_) {
        receives_.insert(std::make_pair(t, std::make_shared<PendingReceives>()));
    }
}

int CudfAllToAll::insert(const std::shared_ptr<cudf::table_view> &table, int32_t target) {
    return insert(table, target, -1);
}

int CudfAllToAll::insert(const std::shared_ptr<cudf::table_view> &tview,
                          int32_t target,
                          int32_t reference) {
    // todo: check weather we have enough memory
    // lets save the table into pending and move on
    makeTableBuffers(tview, target, reference, sendQueues[target]);
    return 1;
}

int CudfAllToAll::insert(cudf::table_view &tview, std::vector<cudf::size_type> &offsets, int ref) {

    // if there is already a partitioned table being sent, return false
    if (ptview)
        return 0;

    ptview = std::make_unique<PartTableView>(tview, offsets);
    for (int i = 0; i < ptview->numberOfParts(); ++i) {
        makePartTableBuffers(i, ref, sendQueues[i]);
    }

    return 1;
}

bool CudfAllToAll::isComplete() {

    if (completed_)
        return true;

    for (auto &pair : sendQueues) {
        // if the buffer queue is not empty, first insert those buffers to a2a
        auto bufferQueue = &(pair.second);
        while (!bufferQueue->empty()) {
            auto pb = bufferQueue->front();
            bool accepted = pb->sendBuffer(all_);
            if (accepted) {
                bufferQueue->pop();
            } else {
                return false;
            }
        }
    }

    if (finished && !finishCalled_) {
        all_->finish();
        finishCalled_ = true;
    }

    if (!all_->isComplete()) {
        return false;
    }

    completed_ = true;
    // all done, reset PartTableView if exists
    if (ptview)
        ptview.reset();

    return true;
}

void CudfAllToAll::finish() {
    finished = true;
}

void CudfAllToAll::close() {
    // clear the input map
    sendQueues.clear();
    // call close on the underlying all-to-all
    all_->close();

    delete allocator_;
}

std::unique_ptr<int []> CudfAllToAll::makeTableHeader(int headersLength,
                                                      int ref,
                                                      int32_t numberOfColumns,
                                                      int numberOfRows) {
    auto tableHeaders = std::make_unique<int32_t []>(headersLength);
    tableHeaders[0] = 0; // shows it is a table header.
    tableHeaders[1] = ref;
    tableHeaders[2] = numberOfColumns;
    tableHeaders[3] = numberOfRows;
    return tableHeaders;
}

std::unique_ptr<int []> CudfAllToAll::makeColumnHeader(int headersLength,
                                                       int columnIndex,
                                                       int typeId,
                                                       bool hasMask,
                                                       bool hasOffset,
                                                       int numberOfElements) {

    auto headers = std::make_unique<int []>(headersLength);
    headers[0] = 1; // shows it is a column header
    headers[1] = columnIndex;
    headers[2] = typeId;
    headers[3] = hasMask;
    headers[4] = hasOffset;
    headers[5] = numberOfElements;
    return headers;
}

void CudfAllToAll::makeTableBuffers(std::shared_ptr<cudf::table_view> table,
                                    int target,
                                    int ref,
                                    std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue) {
    // construct header message to send
    int32_t columns = table->num_columns();
    int32_t headersLength = 4;
    auto tableHeaders = makeTableHeader(headersLength, ref, columns, table->num_rows());
    auto pb = std::make_shared<PendingBuffer>(target, std::move(tableHeaders), headersLength);
    bufferQueue.emplace(pb);

    for (int i = 0; i < columns; ++i) {
        makeColumnBuffers(table->column(i), i, target, bufferQueue);
    }
}

void CudfAllToAll::makePartTableBuffers(int partIndex,
                                        int ref,
                                        std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue) {
    int target = partIndex;

    int columns = ptview->numberOfColumns();
    int headersLength = 4;
    auto tableHeaders = makeTableHeader(headersLength, ref, columns, ptview->numberOfRows(partIndex));
    auto pb = std::make_shared<PendingBuffer>(target, std::move(tableHeaders), headersLength);
    bufferQueue.emplace(pb);

    // if there is zero rows in the partition, no need to send columns
    if (ptview->numberOfRows(partIndex) == 0) {
        return;
    }

    for (int i = 0; i < columns; ++i) {
        makePartColumnBuffers(ptview->column(i), partIndex, i, target, bufferQueue);
    }
}

void CudfAllToAll::makePartColumnBuffers(std::shared_ptr<PartColumnView> pcv,
                                     int partIndex,
                                     int columnIndex,
                                     int target,
                                     std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue) {

    int headersLength = 6;
    auto columnHeaders = makeColumnHeader(headersLength,
                                          columnIndex,
                                          pcv->getColumnTypeId(),
                                          pcv->getColumnView().nullable(),
                                          pcv->getColumnView().num_children(),
                                          pcv->numberOfElements(partIndex));


    auto pb = std::make_shared<PendingBuffer>(pcv->getDataBuffer(partIndex),
                                              pcv->getDataBufferSize(partIndex),
                                              target,
                                              std::move(columnHeaders),
                                              headersLength);
    bufferQueue.emplace(pb);

    if (pcv->getColumnView().nullable()) {
        pb = std::make_shared<PendingBuffer>(pcv->getMaskBuffer(partIndex), pcv->getMaskBufferSize(partIndex), target);
        bufferQueue.emplace(pb);
    }

    if (pcv->getOffsetBufferSize(partIndex) > 0) {
        pb = std::make_shared<PendingBuffer>(pcv->getOffsetBuffer(partIndex),
                                             pcv->getOffsetBufferSize(partIndex),
                                             target);
        bufferQueue.emplace(pb);
    }
}

void CudfAllToAll::makeColumnBuffers(const cudf::column_view &cw,
                                     int columnIndex,
                                     int target,
                                     std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue) {

    // we support uniform size data types and the string type
    if (!uniform_size_data(cw) && cw.type().id() != cudf::type_id::STRING) {
        throw "only uniform-size data-types and the string is supported.";
    }

    int headersLength = 6;
    auto columnHeaders = makeColumnHeader(headersLength,
                                          columnIndex,
                                          (int)(cw.type().id()),
                                          cw.nullable(),
                                          cw.num_children(),
                                          cw.size());

    // insert data buffer
    const uint8_t *dataBuffer;
    int bufferSize;

    // if it is a string column, get char buffer
    const uint8_t *offsetsBuffer;
    int offsetsSize = -1;
    if (cw.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view scv(cw);
        dataBuffer = scv.chars().data<uint8_t>();
        bufferSize = scv.chars_size();

        offsetsBuffer = scv.offsets().data<uint8_t>();
        offsetsSize = dataLength(scv.offsets());
        // get uniform size column data
    } else {
        dataBuffer = cw.data<uint8_t>();
        bufferSize = dataLength(cw);
//    LOG(INFO) << myrank << "******* inserting column buffer with length: " << dataLen;
    }
    // insert the data buffer
    if(bufferSize < 0) {
        throw "bufferSize is negative: " + std::to_string(bufferSize);
    }

    auto pb = std::make_shared<PendingBuffer>(dataBuffer, bufferSize, target, std::move(columnHeaders), headersLength);
    bufferQueue.emplace(pb);

    // insert null buffer if exists
    if (cw.nullable()) {
        uint8_t * nullBuffer = (uint8_t *)cw.null_mask();
        std::size_t nullBufSize = cudf::bitmask_allocation_size_bytes(cw.size());
        pb = std::make_shared<PendingBuffer>(nullBuffer, nullBufSize, target);
        bufferQueue.emplace(pb);
    }

    if (offsetsSize >= 0) {
        pb = std::make_shared<PendingBuffer>(offsetsBuffer, offsetsSize, target);
        bufferQueue.emplace(pb);
    }
}

void CudfAllToAll::constructColumn(std::shared_ptr<PendingReceives> pr) {

    std::unique_ptr<cudf::column> column;

    cudf::data_type dt(static_cast<cudf::type_id>(pr->columnDataType));
    std::shared_ptr<rmm::device_buffer> dataBuffer = pr->dataBuffer;
    std::shared_ptr<rmm::device_buffer> nullBuffer = pr->nullBuffer;
    std::shared_ptr<rmm::device_buffer> offsetsBuffer = pr->offsetsBuffer;

    if (dt.id() != cudf::type_id::STRING)  {
        if(pr->hasNullBuffer) {
            column = std::make_unique<cudf::column>(dt,
                                                    pr->dataSize,
                                                    std::move(*dataBuffer),
                                                    std::move(*nullBuffer));
        } else {
            column = std::make_unique<cudf::column>(dt, pr->dataSize, std::move(*dataBuffer));
        }

    // construct string column
    } else {
        // construct chars child column
        auto cdt = cudf::data_type{cudf::type_id::INT8};
        auto charsColumn = std::make_unique<cudf::column>(cdt, pr->dataBufferLen, std::move(*dataBuffer));

        int32_t offBase = getScalar<int32_t>((uint8_t *)offsetsBuffer->data());
        // todo: can offsets start from non zero values in non-partitioned tables
        //       we need to make sure of this
        if (offBase > 0) {
            callRebaseOffsets((int32_t *)offsetsBuffer->data(), pr->dataSize + 1, offBase);
        }

        auto odt = cudf::data_type{cudf::type_id::INT32};
        auto offsetsColumn = std::make_unique<cudf::column>(odt, pr->dataSize + 1, std::move(*offsetsBuffer));

        // this creates a new buffer, so less efficient
        // int32_t offsetBase = getScalar<int32_t>((uint8_t *)offsetsBuffer->data());
//        if (offsetBase > 0) {
//            auto base = std::make_unique<cudf::numeric_scalar<int32_t>>(offsetBase, true);
//            offsetsColumn =
//                cudf::binary_operation(offsetsColumn->view(), *base, cudf::binary_operator::SUB, offsetsColumn->type());
//            cudaDeviceSynchronize();
//        }

        std::vector<std::unique_ptr<cudf::column>> children;
        children.emplace_back(std::move(offsetsColumn));
        children.emplace_back(std::move(charsColumn));

        if (pr->hasNullBuffer) {
            rmm::device_buffer rmmBuf{0, rmm::cuda_stream_default, rmm::mr::get_current_device_resource()};
            column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                    pr->dataSize,
                                                    std::move(rmmBuf),
                                                    std::move(*nullBuffer),
                                                    cudf::UNKNOWN_NULL_COUNT,
                                                    std::move(children));
        } else{
            column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                    pr->dataSize,
                                                    std::move(rmm::device_buffer{0, rmm::cuda_stream_default}),
                                                    std::move(rmm::device_buffer{0, rmm::cuda_stream_default}),
                                                    0,
                                                    std::move(children));
        }

    }

    // if the column is constructed, add it to the list
    if(column) {
        pr->columns.insert({pr->columnIndex, std::move(column)});

        // clear column related data from pr
        pr->columnIndex = -1;
        pr->columnDataType = -1;
        pr->dataSize = 0;
        pr->dataBuffer.reset();
        pr->nullBuffer.reset();
        pr->offsetsBuffer.reset();
        pr->hasNullBuffer = false;
        pr->hasOffsetBuffer = false;
        pr->dataBufferLen = 0;
    }
}

std::shared_ptr<cudf::table> CudfAllToAll::constructTable(std::shared_ptr<PendingReceives> pr) {

    std::vector<std::unique_ptr<cudf::column>> columnVector{};
    for (long unsigned int i=0; i < pr->columns.size(); i++) {
        columnVector.push_back(std::move(pr->columns.at(i)));
    }

    return std::make_shared<cudf::table>(std::move(columnVector));
}

/**
 * This function is called when a data is received
 */
bool CudfAllToAll::onReceive(int source, std::shared_ptr<cylon::Buffer> buffer, int length) {
//  LOG(INFO) << myrank << ",,,,, buffer received from the source: " << source << ", with length: " << length;
  if (length == 0) {
      return true;
  }
  std::shared_ptr<CudfBuffer> cb = std::dynamic_pointer_cast<CudfBuffer>(buffer);

  // if the data buffer is not received yet, get it
  std::shared_ptr<PendingReceives> pr = receives_.at(source);
  if (!pr->dataBuffer) {
    pr->dataBuffer = cb->getBuf();
    pr->dataBufferLen = length;

    // if there is no null buffer or offset buffer, create the column
    if(!pr->hasNullBuffer && !pr->hasOffsetBuffer) {
        constructColumn(pr);
    }
  } else if(pr->hasNullBuffer && !pr->nullBuffer) {
      pr->nullBuffer = cb->getBuf();
      // if there is no offset buffer, create the column
      if (!pr->hasOffsetBuffer) {
          constructColumn(pr);
      }
  } else if(pr->hasOffsetBuffer && !pr->offsetsBuffer) {
      pr->offsetsBuffer = cb->getBuf();
      constructColumn(pr);
  } else {
      LOG(WARNING) << myrank <<  " columnIndex: "<< pr->columnIndex << " an unexpected buffer received from: " << source
            << ", buffer length: " << length;
      return false;
  }

  // if all columns are created, create the table
  if ((int32_t)pr->columns.size() == pr->numberOfColumns) {
      std::shared_ptr<cudf::table> tbl = constructTable(pr);
      recv_callback_(source, tbl, pr->reference);

      // clear table data from pr
      pr->columns.clear();
      pr->numberOfColumns = -1;
      pr->reference = -1;
  }

  return true;
}

/**
 * Receive the header, this happens before we receive the actual data
 */
bool CudfAllToAll::onReceiveHeader(int source, int finished, int *buffer, int length) {
  if (length > 0) {
      if (buffer[0] == 0) { // table header
          // if the incoming table has zero rows, if it is empty
          // nothing to be done, no data will be received for this table.
          // ignore the message
          if (buffer[3] == 0) {
              return true;
          }
          std::shared_ptr<PendingReceives> pr = receives_.at(source);
          pr->reference = buffer[1];
          pr->numberOfColumns = buffer[2];
      } else if(buffer[0] == 1){ // column header
          std::shared_ptr<PendingReceives> pr = receives_.at(source);
          pr->columnIndex = buffer[1];
          pr->columnDataType = buffer[2];
          pr->hasNullBuffer = buffer[3];
          pr->hasOffsetBuffer = buffer[4];
          pr->dataSize = buffer[5];
//          LOG(INFO) << myrank << "----received a column header from the source: " << source
//                  << ", columnIndex: " << pr->columnIndex << std::endl
//                  << ", columnDataType: " << pr->columnDataType << std::endl
//                  << ", hasNullBuffer: " << pr->hasNullBuffer << std::endl
//                  << ", hasOffsetBuffer: " << pr->hasOffsetBuffer << std::endl
//                  << ", dataSize: " << pr->dataSize << std::endl;
      }
  }
  return true;
}

/**
 * This method is called after we successfully send a buffer
 * @return
 */
bool CudfAllToAll::onSendComplete(int target, const void *buffer, int length) {
//  LOG(INFO) << myrank << ", SendComplete with length: " << length << " for the target: " << target;
  return true;
}

}// end of namespace gcylon
