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
#include <gcylon/utils/util.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/null_mask.hpp>

namespace gcylon {

CudfTableSerializer::CudfTableSerializer(const cudf::table_view &tv) : tv_(tv) {
}

void CudfTableSerializer::initTableBuffers() {
  if(tv_.num_rows() == 0) {
    buffer_sizes_ = getEmptyTableBufferSizes();
    table_buffers_.resize(getNumberOfBuffers(), nullptr);
    table_buffers_initialized = true;
    return;
  }

  // for each column, we keep 3 data objects: data, mask, offsets
  std::pair<int32_t, const uint8_t *> p;
  for (int i = 0; i < tv_.num_columns(); ++i) {
    auto cv = tv_.column(i);
    p = getColumnData(cv);
    buffer_sizes_.push_back(p.first);
    table_buffers_.push_back(p.second);

    p = getColumnMask(cv);
    buffer_sizes_.push_back(p.first);
    table_buffers_.push_back(p.second);

    p = getColumnOffsets(cv);
    buffer_sizes_.push_back(p.first);
    table_buffers_.push_back(p.second);
  }
  table_buffers_initialized = true;
}

const std::vector<int32_t> &CudfTableSerializer::getBufferSizes() {
  if (!table_buffers_initialized) {
    initTableBuffers();
  }

  return buffer_sizes_;
}

const std::vector<const uint8_t *> &CudfTableSerializer::getDataBuffers() {
  if (!table_buffers_initialized) {
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

std::pair<int32_t, const uint8_t *> CudfTableSerializer::getColumnData(const cudf::column_view &cv) {
  if (cv.type().id() == cudf::type_id::STRING) {
    cudf::strings_column_view scv(cv);
    int32_t offset_in_bytes = 0;
    int32_t end_in_bytes = scv.chars_size();
    if (scv.offset() > 0) {
      // if the column view is a table slice that does not start from the first row
      // get the offset from the offsets column
      offset_in_bytes = getScalar<int32_t>(scv.offsets().head<int32_t>() + scv.offset());
    }

    if (scv.size() < scv.offsets().size() - 1) {
      // if the column view is a table slice that does not end at the last row
      // get the slice end from the offsets column
      end_in_bytes = getScalar<int32_t>(scv.offsets().head<int32_t>() + scv.offset() + scv.size());
    }
    int32_t size_in_bytes = end_in_bytes - offset_in_bytes;
    return std::make_pair(size_in_bytes,
                          static_cast<const uint8_t *>(scv.chars().head<uint8_t>()) + offset_in_bytes);
  }

  int32_t size_in_bytes = cudf::size_of(cv.type()) * cv.size();
  int32_t offset_in_bytes = cudf::size_of(cv.type()) * cv.offset();
  return std::make_pair(size_in_bytes, static_cast<const uint8_t *>(cv.head<uint8_t>()) + offset_in_bytes);
}

std::pair<int32_t, const uint8_t *> CudfTableSerializer::getColumnMask(const cudf::column_view &cv) {
  if (cv.has_nulls()) {
    if (cv.offset() == 0) {
      int32_t size = cudf::bitmask_allocation_size_bytes(cv.size());
      return std::make_pair(size, reinterpret_cast<const uint8_t *>(cv.null_mask()));
    } else if (cv.offset() % 8 == 0) {
      // if the offset is a multiple of 8, there is no need for copying the null mask
      // we can ignore the first offset_size/8 bytes and send the remaining null mask buffer
      int offset_start_in_bytes = cv.offset() / 8;
      int32_t size = cudf::bitmask_allocation_size_bytes(cv.size()) - offset_start_in_bytes;
      return std::make_pair(size, reinterpret_cast<const uint8_t *>(cv.null_mask()) + offset_start_in_bytes);
    } else {
      auto mask_buf = cudf::copy_bitmask(cv.null_mask(), cv.offset(), cv.offset() + cv.size());
      rmm::cuda_stream_default.synchronize();
      auto buff = static_cast<const uint8_t *>(mask_buf.data());
      int32_t size = mask_buf.size();
      mask_buffers.push_back(std::move(mask_buf));
      return std::make_pair(size, buff);
    }
  }

  return std::make_pair(0, nullptr);
}

std::pair<int32_t, const uint8_t *> CudfTableSerializer::getColumnOffsets(const cudf::column_view &cv) {
  if (cv.type().id() == cudf::type_id::STRING) {
    cudf::strings_column_view scv(cv);
    int size_in_bytes = cudf::size_of(scv.offsets().type()) * (scv.size() + 1);
    int offset_in_bytes = cudf::size_of(scv.offsets().type()) * scv.offset();
    return std::make_pair(size_in_bytes,
                          static_cast<const uint8_t *>(scv.offsets().head<uint8_t>()) + offset_in_bytes);
  }

  return std::make_pair(0, nullptr);
}

std::vector<int32_t> CudfTableSerializer::getDataTypes() {
  std::vector<int32_t> data_types;
  for (int i = 0; i < tv_.num_columns(); ++i) {
    data_types.push_back(static_cast<int32_t>(tv_.column(i).type().id()));
  }
  return data_types;
}

} // end of namespace gcylon