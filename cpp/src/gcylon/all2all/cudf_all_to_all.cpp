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
#include <cudf/copying.hpp>
#include <cudf/concatenate.hpp>

#include <gcylon/all2all/cudf_all_to_all.hpp>
#include <gcylon/utils/util.hpp>
#include <gcylon/all2all/cudf_all_to_all.cuh>
#include <gcylon/net/cudf_net_ops.hpp>

#include <cylon/util/macros.hpp>
#include <cylon/net/mpi/mpi_communicator.hpp>

cylon::Status gcylon::net::AllToAll(const cudf::table_view &tv,
                                    const std::vector<cudf::size_type> &part_indices,
                                    const std::shared_ptr<cylon::CylonContext> &ctx,
                                    std::vector<std::unique_ptr<cudf::table>> &received_tables) {

  if (part_indices.size() != ctx->GetWorldSize() + 1) {
    return cylon::Status(cylon::Code::ValueError,
                         "there must be number of workers + 1 values in part_indices");
  }

  const auto &neighbours = ctx->GetNeighbours(true);
  received_tables.resize(ctx->GetWorldSize());

  // define call back to catch the receiving tables
  CudfCallback cudf_callback =
    [&received_tables](int source, std::unique_ptr<cudf::table> received_table, int reference) {
      if (received_tables[source] != nullptr) {
        throw std::string("More than one table is received from the worker: ") + std::to_string(source);
      }
      received_tables[source] = std::move(received_table);
      return true;
    };

  // doing all to all communication to exchange tables
  CudfAllToAll all_to_all(ctx, neighbours, neighbours, ctx->GetNextSequence(), std::move(cudf_callback));

  // insert partitioned table for all-to-all
  int accepted = all_to_all.insert(tv, part_indices, ctx->GetNextSequence());
  if (!accepted) {
    return cylon::Status(accepted);
  }

  // wait for the partitioned tables to arrive
  // now complete the communication
  all_to_all.finish();
  while (!all_to_all.isComplete()) {}
  all_to_all.close();

  // if no table is received from any worker,
  // create an empty table for those
  for (int i = 0; i < received_tables.size(); ++i) {
    if (received_tables[i] == nullptr) {
      received_tables[i] = cudf::empty_like(tv);
    }
  }

  return cylon::Status::OK();
}

cylon::Status gcylon::net::AllToAll(const cudf::table_view &tv,
                                    const std::vector<cudf::size_type> &part_indices,
                                    const std::shared_ptr<cylon::CylonContext> &ctx,
                                    std::unique_ptr<cudf::table> &table_out) {

  std::vector<std::unique_ptr<cudf::table>> received_tables;
  received_tables.reserve(ctx->GetWorldSize());

  RETURN_CYLON_STATUS_IF_FAILED(
    gcylon::net::AllToAll(tv, part_indices, ctx, received_tables));

  std::vector<cudf::table_view> views = gcylon::tablesToViews(received_tables);
  table_out = cudf::concatenate(views);
  return cylon::Status::OK();
}

namespace gcylon {

//////////////////////////////////////////////////////////////////////
// global types and fuctions
//////////////////////////////////////////////////////////////////////

/**
* data buffer length of a column in bytes
* @param input
* @return
*/
cudf::size_type dataLength(cudf::column_view const &cw) {
  // even null values exist in the buffer with unspecified values
  return cudf::size_of(cw.type()) * cw.size();
}

//////////////////////////////////////////////////////////////////////
// PendingBuffer implementations
//////////////////////////////////////////////////////////////////////
PendingBuffer::PendingBuffer(const uint8_t *buffer,
                             int buffer_size,
                             int target,
                             std::unique_ptr<int[]> headers,
                             int headers_length) :
    buffer(buffer),
    buffer_size(buffer_size),
    target(target),
    headers(std::move(headers)),
    headers_length(headers_length) {}

PendingBuffer::PendingBuffer(int target,
                             std::unique_ptr<int[]> headers,
                             int headers_length) :
    buffer(nullptr),
    buffer_size(-1),
    target(target),
    headers(std::move(headers)),
    headers_length(headers_length) {}

bool PendingBuffer::sendBuffer(std::shared_ptr<cylon::AllToAll> all) {
  // if there is no data buffer, only header buffer
  if (buffer_size <= 0) {
    bool accepted = all->insert(nullptr, 0, target, headers.get(), headers_length);
    if (!accepted) {
      LOG(WARNING) << " header buffer not accepted to be sent";
    }
    return accepted;
  }

  // if there is no header buffer, only data buffer
  if (headers_length < 0) {
    bool accepted = all->insert(buffer, buffer_size, target);
    if (!accepted) {
      LOG(WARNING) << " data buffer not accepted to be sent";
    }
    return accepted;
  }

  bool accepted = all->insert(buffer, buffer_size, target, headers.get(), headers_length);
  if (!accepted) {
    LOG(WARNING) << " data buffer with header not accepted to be sent";
  }
  return accepted;
}

//////////////////////////////////////////////////////////////////////
// PartColumnView implementations
//////////////////////////////////////////////////////////////////////
PartColumnView::PartColumnView(const cudf::column_view &cv, const std::vector<cudf::size_type> &part_indexes)
    : cv(cv), part_indexes(part_indexes), part_char_offsets(part_indexes.size()) {

  if (cv.type().id() == cudf::type_id::STRING) {
    scv = std::make_unique<cudf::strings_column_view>(this->cv);

    // get offsets from gpu to cpu concurrently
    int offset_data_type_size = cudf::size_of(scv->offsets().type());
    uint8_t *dest = (uint8_t *) part_char_offsets.data();
    const uint8_t *src = scv->offsets().data<uint8_t>();
    for (long unsigned int i = 0; i < part_indexes.size(); ++i) {
      cudaMemcpyAsync(dest + offset_data_type_size * i,
                      src + offset_data_type_size * part_indexes[i],
                      offset_data_type_size,
                      cudaMemcpyDeviceToHost);
    }
    // synch on the default stream
    cudaStreamSynchronize(0);
  }

  if (cv.nullable()) {
    for (long unsigned int i = 0; i < part_indexes.size() - 1; ++i) {
      auto mask_buf = cudf::copy_bitmask(cv.null_mask(), part_indexes[i], part_indexes[i + 1]);
      mask_buffers.emplace(std::make_pair(i, std::move(mask_buf)));
    }
    rmm::cuda_stream_default.synchronize();
  }
}

const uint8_t *PartColumnView::getDataBuffer(int part_index) {
  if (cv.type().id() == cudf::type_id::STRING) {
    return scv->chars().data<uint8_t>() + part_char_offsets[part_index];
  }

  int start_pos = cudf::size_of(cv.type()) * part_indexes[part_index];
  return cv.data<uint8_t>() + start_pos;
}

int PartColumnView::getDataBufferSize(int part_index) {
  if (cv.type().id() == cudf::type_id::STRING) {
    return part_char_offsets[part_index + 1] - part_char_offsets[part_index];
  }

  return cudf::size_of(cv.type()) * numberOfElements(part_index);
}

const uint8_t *PartColumnView::getOffsetBuffer(int part_index) {
  if (cv.type().id() == cudf::type_id::STRING) {
    return scv->offsets().data<uint8_t>() + part_indexes[part_index] * cudf::size_of(scv->offsets().type());
  }

  return nullptr;
}

int PartColumnView::getOffsetBufferSize(int part_index) {
  if (cv.type().id() == cudf::type_id::STRING) {
    if (numberOfElements(part_index) == 0) {
      return 0;
    } else {
      return (numberOfElements(part_index) + 1) * cudf::size_of(scv->offsets().type());
    }
  }

  return 0;
}

const uint8_t *PartColumnView::getMaskBuffer(int part_index) {
  if (!cv.nullable()) {
    return nullptr;
  }
  return (uint8_t *) mask_buffers.at(part_index).data();
}

int PartColumnView::getMaskBufferSize(int part_index) {
  if (!cv.nullable())
    return 0;
  return mask_buffers.at(part_index).size();
}

//////////////////////////////////////////////////////////////////////
// PartTableView implementations
//////////////////////////////////////////////////////////////////////
PartTableView::PartTableView(const cudf::table_view &tv, const std::vector<cudf::size_type> &part_indexes)
    : tv(tv), part_indexes(part_indexes) {

  for (int i = 0; i < this->tv.num_columns(); ++i) {
    auto pcv = std::make_shared<PartColumnView>(this->tv.column(i), this->part_indexes);
    columns.insert(std::make_pair(i, pcv));
  }
}

std::shared_ptr<PartColumnView> PartTableView::column(int column_index) {
  return columns.at(column_index);
}

//////////////////////////////////////////////////////////////////////
// CudfAllToAll implementations
//////////////////////////////////////////////////////////////////////
CudfAllToAll::CudfAllToAll(std::shared_ptr<cylon::CylonContext> ctx,
                           const std::vector<int> &sources,
                           const std::vector<int> &targets,
                           int edge_id,
                           CudfCallback callback) :
    rank_(ctx->GetRank()),
    sources_(sources),
    targets_(targets),
    recv_callback_(std::move(callback)) {

  allocator_ = new CudfAllocator();

  // we need to pass the correct arguments
  all_ = std::make_shared<cylon::AllToAll>(ctx, sources_, targets_, edge_id, this, allocator_);

  // add the trackers for sending
  for (auto t: targets_) {
    send_queues_.insert(std::make_pair(t, std::queue<std::shared_ptr<PendingBuffer>>()));
  }

  for (auto t: sources_) {
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
  makeTableBuffers(tview, target, reference, send_queues_[target]);
  return 1;
}

int CudfAllToAll::insert(const cudf::table_view &tview, const std::vector<cudf::size_type> &offsets, int ref) {

  // if there is already a partitioned table being sent, return false
  if (ptview_)
    return 0;

  ptview_ = std::make_unique<PartTableView>(tview, offsets);
  for (int i = 0; i < ptview_->numberOfParts(); ++i) {
    makePartTableBuffers(i, ref, send_queues_[i]);
  }

  return 1;
}

bool CudfAllToAll::isComplete() {

  if (completed_)
    return true;

  for (auto &pair: send_queues_) {
    // if the buffer queue is not empty, first insert those buffers to a2a
    auto buffer_queue = &(pair.second);
    while (!buffer_queue->empty()) {
      auto pb = buffer_queue->front();
      bool accepted = pb->sendBuffer(all_);
      if (accepted) {
        buffer_queue->pop();
      } else {
        return false;
      }
    }
  }

  if (finished_ && !finish_called_) {
    all_->finish();
    finish_called_ = true;
  }

  if (!all_->isComplete()) {
    return false;
  }

  completed_ = true;
  // all done, reset PartTableView if exists
  if (ptview_)
    ptview_.reset();

  return true;
}

void CudfAllToAll::finish() {
  finished_ = true;
}

void CudfAllToAll::close() {
  // clear the input map
  send_queues_.clear();
  // call close on the underlying all-to-all
  all_->close();

  delete allocator_;
}

std::unique_ptr<int[]> CudfAllToAll::makeTableHeader(int headers_length,
                                                     int ref,
                                                     int32_t number_of_columns,
                                                     int number_of_rows) {
  auto table_headers = std::make_unique<int32_t[]>(headers_length);
  table_headers[0] = 0; // shows it is a table header.
  table_headers[1] = ref;
  table_headers[2] = number_of_columns;
  table_headers[3] = number_of_rows;
  return table_headers;
}

std::unique_ptr<int[]> CudfAllToAll::makeColumnHeader(int headers_length,
                                                      int column_index,
                                                      bool has_data_buffer,
                                                      bool has_mask,
                                                      bool has_offset,
                                                      int number_of_elements) {

  auto headers = std::make_unique<int[]>(headers_length);
  headers[0] = 1; // shows it is a column header
  headers[1] = column_index;
  headers[2] = has_data_buffer;
  headers[3] = has_mask;
  headers[4] = has_offset;
  headers[5] = number_of_elements;
  return headers;
}

void CudfAllToAll::makeTableBuffers(std::shared_ptr<cudf::table_view> table,
                                    int target,
                                    int ref,
                                    std::queue<std::shared_ptr<PendingBuffer>> &buffer_queue) {
  // construct header message to send
  int32_t columns = table->num_columns();
  int32_t headers_length = 4;
  auto table_headers = makeTableHeader(headers_length, ref, columns, table->num_rows());
  auto pb = std::make_shared<PendingBuffer>(target, std::move(table_headers), headers_length);
  buffer_queue.emplace(pb);

  for (int i = 0; i < columns; ++i) {
    makeColumnBuffers(table->column(i), i, target, buffer_queue);
  }
}

void CudfAllToAll::makePartTableBuffers(int part_index,
                                        int ref,
                                        std::queue<std::shared_ptr<PendingBuffer>> &buffer_queue) {
  int target = part_index;

  int columns = ptview_->numberOfColumns();
  int headers_length = 4;
  auto table_headers = makeTableHeader(headers_length, ref, columns, ptview_->numberOfRows(part_index));
  auto pb = std::make_shared<PendingBuffer>(target, std::move(table_headers), headers_length);
  buffer_queue.emplace(pb);

  // if there is zero rows in the partition, no need to send columns
  if (ptview_->numberOfRows(part_index) == 0) {
    return;
  }

  for (int i = 0; i < columns; ++i) {
    makePartColumnBuffers(ptview_->column(i), part_index, i, target, buffer_queue);
  }
}

void CudfAllToAll::makePartColumnBuffers(std::shared_ptr<PartColumnView> pcv,
                                         int part_index,
                                         int column_index,
                                         int target,
                                         std::queue<std::shared_ptr<PendingBuffer>> &buffer_queue) {

  int headers_length = 6;
  auto column_headers = makeColumnHeader(headers_length,
                                         column_index,
                                         pcv->getDataBufferSize(part_index),
                                         pcv->getColumnView().nullable(),
                                         pcv->getColumnView().num_children(),
                                         pcv->numberOfElements(part_index));


  auto pb = std::make_shared<PendingBuffer>(pcv->getDataBuffer(part_index),
                                            pcv->getDataBufferSize(part_index),
                                            target,
                                            std::move(column_headers),
                                            headers_length);
  buffer_queue.emplace(pb);

  if (pcv->getColumnView().nullable()) {
    pb = std::make_shared<PendingBuffer>(pcv->getMaskBuffer(part_index), pcv->getMaskBufferSize(part_index),
                                         target);
    buffer_queue.emplace(pb);
  }

  if (pcv->getOffsetBufferSize(part_index) > 0) {
    pb = std::make_shared<PendingBuffer>(pcv->getOffsetBuffer(part_index),
                                         pcv->getOffsetBufferSize(part_index),
                                         target);
    buffer_queue.emplace(pb);
  }
}

void CudfAllToAll::makeColumnBuffers(const cudf::column_view &cw,
                                     int column_index,
                                     int target,
                                     std::queue<std::shared_ptr<PendingBuffer>> &buffer_queue) {

  // we support uniform size data types and the string type
  if (!cudf::is_fixed_width(cw.type()) && cw.type().id() != cudf::type_id::STRING) {
    throw "only fixed-width data-types and the string are supported.";
  }

  // insert data buffer
  const uint8_t *data_buffer;
  int buffer_size;

  // if it is a string column, get char buffer
  const uint8_t *offsets_buffer;
  int offsets_size = -1;
  if (cw.type().id() == cudf::type_id::STRING) {
    cudf::strings_column_view scv(cw);
    data_buffer = scv.chars().data<uint8_t>();
    buffer_size = scv.chars_size();

    offsets_buffer = scv.offsets().data<uint8_t>();
    offsets_size = dataLength(scv.offsets());
    // get uniform size column data
  } else {
    data_buffer = cw.data<uint8_t>();
    buffer_size = dataLength(cw);
  }
  // insert the data buffer
  if (buffer_size < 0) {
    throw "buffer_size is negative: " + std::to_string(buffer_size);
  }

  int headers_length = 6;
  auto column_headers = makeColumnHeader(headers_length,
                                         column_index,
                                         buffer_size,
                                         cw.nullable(),
                                         cw.num_children(),
                                         cw.size());

  auto pb = std::make_shared<PendingBuffer>(data_buffer, buffer_size, target, std::move(column_headers),
                                            headers_length);
  buffer_queue.emplace(pb);

  // insert null buffer if exists
  if (cw.nullable()) {
    uint8_t *null_buffer = (uint8_t *) cw.null_mask();
    std::size_t null_buf_size = cudf::bitmask_allocation_size_bytes(cw.size());
    pb = std::make_shared<PendingBuffer>(null_buffer, null_buf_size, target);
    buffer_queue.emplace(pb);
  }

  if (offsets_size >= 0) {
    pb = std::make_shared<PendingBuffer>(offsets_buffer, offsets_size, target);
    buffer_queue.emplace(pb);
  }
}

void CudfAllToAll::constructColumn(std::shared_ptr<PendingReceives> pr) {

  std::unique_ptr<cudf::column> column;

  cudf::data_type dt = ptview_->column(pr->column_index)->getColumnDataType();
  std::shared_ptr<rmm::device_buffer> data_buffer = pr->data_buffer;
  std::shared_ptr<rmm::device_buffer> null_buffer = pr->null_buffer;
  std::shared_ptr<rmm::device_buffer> offsets_buffer = pr->offsets_buffer;

  if (dt.id() != cudf::type_id::STRING) {
    if (pr->has_null_buffer) {
      column = std::make_unique<cudf::column>(dt,
                                              pr->data_size,
                                              std::move(*data_buffer),
                                              std::move(*null_buffer));
    } else {
      column = std::make_unique<cudf::column>(dt, pr->data_size, std::move(*data_buffer));
    }

    // construct string column
  } else {
    // construct chars child column
    auto cdt = cudf::data_type{cudf::type_id::INT8};
    auto chars_column = std::make_unique<cudf::column>(cdt, pr->data_buffer_len, std::move(*data_buffer));

    int32_t off_base = getScalar((int32_t *) offsets_buffer->data());
    if (off_base > 0) {
      callRebaseOffsets((int32_t *) offsets_buffer->data(), pr->data_size + 1, off_base);
    }

    auto odt = cudf::data_type{cudf::type_id::INT32};
    auto offsets_column = std::make_unique<cudf::column>(odt, pr->data_size + 1, std::move(*offsets_buffer));

    std::vector<std::unique_ptr<cudf::column>> children;
    children.emplace_back(std::move(offsets_column));
    children.emplace_back(std::move(chars_column));

    if (pr->has_null_buffer) {
      rmm::device_buffer rmm_buf{0, rmm::cuda_stream_default, rmm::mr::get_current_device_resource()};
      column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                              pr->data_size,
                                              std::move(rmm_buf),
                                              std::move(*null_buffer),
                                              cudf::UNKNOWN_NULL_COUNT,
                                              std::move(children));
    } else {
      column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                              pr->data_size,
                                              std::move(rmm::device_buffer{0, rmm::cuda_stream_default}),
                                              std::move(rmm::device_buffer{0, rmm::cuda_stream_default}),
                                              0,
                                              std::move(children));
    }

  }

  // if the column is constructed, add it to the list
  if (column) {
    pr->columns.push_back(std::move(column));

    // clear column related data from pr
    pr->column_index = -1;
    pr->has_data_buff = false;
    pr->data_size = 0;
    pr->data_buffer.reset();
    pr->null_buffer.reset();
    pr->offsets_buffer.reset();
    pr->has_null_buffer = false;
    pr->has_offset_buffer = false;
    pr->data_buffer_len = 0;
  }
}

std::unique_ptr<cudf::table> CudfAllToAll::constructTable(std::shared_ptr<PendingReceives> pr) {
  return std::make_unique<cudf::table>(std::move(pr->columns));
}

/**
* This function is called when a data is received
*/
bool CudfAllToAll::onReceive(int source, std::shared_ptr<cylon::Buffer> buffer, int length) {

  if (length == 0) {
    return true;
  }
  std::shared_ptr<CudfBuffer> cb = std::dynamic_pointer_cast<CudfBuffer>(buffer);

  // if the data buffer is not received yet, get it
  std::shared_ptr<PendingReceives> pr = receives_.at(source);
  if (pr->has_data_buff && !pr->data_buffer) {
    pr->data_buffer = cb->getBuf();
    pr->data_buffer_len = length;

    // if there is no null buffer or offset buffer, create the column
    if (!pr->has_null_buffer && !pr->has_offset_buffer) {
      constructColumn(pr);
    }
  } else if (pr->has_null_buffer && !pr->null_buffer) {
    pr->null_buffer = cb->getBuf();
    // if there is no offset buffer, create the column
    if (!pr->has_offset_buffer) {
      constructColumn(pr);
    }
  } else if (pr->has_offset_buffer && !pr->offsets_buffer) {
    pr->offsets_buffer = cb->getBuf();
    constructColumn(pr);
  } else {
    LOG(WARNING) << rank_ << " column_index: " << pr->column_index << " an unexpected buffer received from: "
                 << source << ", buffer length: " << length;
    return false;
  }

  // if all columns are created, create the table
  if ((int32_t) pr->columns.size() == pr->number_of_columns) {
    std::unique_ptr<cudf::table> tbl = constructTable(pr);
    recv_callback_(source, std::move(tbl), pr->reference);

    // clear table data from pr
    pr->columns.clear();
    pr->number_of_columns = -1;
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
      pr->number_of_columns = buffer[2];
    } else if (buffer[0] == 1) { // column header
      std::shared_ptr<PendingReceives> pr = receives_.at(source);
      pr->column_index = buffer[1];
      pr->has_data_buff = buffer[2];
      pr->has_null_buffer = buffer[3];
      pr->has_offset_buffer = buffer[4];
      pr->data_size = buffer[5];
      if (!pr->has_data_buff) {
        pr->data_buffer_len = 0;
        pr->data_buffer = std::make_shared<rmm::device_buffer>(0, rmm::cuda_stream_default);
      }
    }
  }
  return true;
}

/**
* This method is called after we successfully send a buffer
* @return
*/
bool CudfAllToAll::onSendComplete(int target, const void *buffer, int length) {
  return true;
}

}// end of namespace gcylon
