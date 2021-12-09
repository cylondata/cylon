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

#include "table_serialize.hpp"

#include <utility>
#include <arrow/util/bit_util.h>
#include <arrow/util/bitmap_ops.h>

namespace cylon {

class CylonTableSerializer : public TableSerializer {
 public:
  CylonTableSerializer(std::shared_ptr<arrow::Table> table,
                       std::vector<int32_t> buffer_sizes,
                       std::vector<const uint8_t *> data_buffers) :
      table_(std::move(table)),
      num_buffers_(table_->num_columns() * 3),
      buffer_sizes_(std::move(buffer_sizes)),
      data_buffers_(std::move(data_buffers)) {
    assert(num_buffers_ == (int) buffer_sizes.size());
    assert(num_buffers_ == (int) data_buffers_.size());
  }

  const std::vector<int32_t> &getBufferSizes() override {
    return buffer_sizes_;
  }
  int getNumberOfBuffers() override {
    return num_buffers_;
  }
  std::vector<int32_t> getEmptyTableBufferSizes() override {
    return std::vector<int32_t>(num_buffers_, 0);
  }
  const std::vector<const uint8_t *> &getDataBuffers() override {
    return data_buffers_;
  }

  std::vector<int32_t> getDataTypes() override {
    std::vector<int32_t> data_types;
    data_types.reserve(table_->num_columns());
    for (const auto &f: table_->schema()->fields()) {
      data_types.push_back(static_cast<int32_t>(tarrow::ToCylonTypeId(f->type())));
    }
    return data_types;
  }

 private:
  const std::shared_ptr<arrow::Table> table_;
  const int32_t num_buffers_;
  const std::vector<int32_t> buffer_sizes_;
  const std::vector<const uint8_t *> data_buffers_;
};

template<int buf_idx = 0>
Status CollectBitmapInfo(const arrow::ArrayData &data, std::vector<int32_t> *buffer_sizes,
                         std::vector<const uint8_t *> *data_buffers,
                         arrow::BufferVector *bitmaps_with_offset,
                         arrow::MemoryPool *pool) {
  if (!data.MayHaveNulls()) {
    buffer_sizes->push_back(0);
    data_buffers->push_back(nullptr);
    return Status::OK();
  }

  // there are nulls
  buffer_sizes->push_back((int32_t) arrow::BitUtil::BytesForBits(data.length));
  if (data.offset == 0) { // no offset
    data_buffers->push_back(data.buffers[buf_idx]->data());
  } else if (data.offset % CHAR_BIT == 0) { // offset is at a byte boundary
    data_buffers->push_back(data.buffers[buf_idx]->data() + data.offset / CHAR_BIT);
  } else { // non-byte boundary offset
    CYLON_ASSIGN_OR_RAISE(
        auto buf,
        arrow::internal::CopyBitmap(pool, data.buffers[buf_idx]->data(), data.offset, data.length))
    data_buffers->push_back(buf->data());
    bitmaps_with_offset->push_back(std::move(buf));
  }

  return Status::OK();
}

Status CollectDataBuffer(const arrow::ArrayData &data, std::vector<int32_t> *buffer_sizes,
                         std::vector<const uint8_t *> *data_buffers,
                         arrow::BufferVector *bitmaps_with_offset,
                         arrow::MemoryPool *pool) {
  const auto &type = data.type;
  if (type->id() == arrow::Type::BOOL) {
    return CollectBitmapInfo<1>(data, buffer_sizes, data_buffers, bitmaps_with_offset, pool);
  }

  if (arrow::is_fixed_width(type->id())) {
    int byte_width = std::static_pointer_cast<arrow::FixedWidthType>(type)->bit_width() / CHAR_BIT;
    buffer_sizes->push_back(byte_width * (int) data.length);
    data_buffers->push_back(data.GetValues<uint8_t>(1));
    return Status::OK();
  }

  if (arrow::is_binary_like(type->id())) {
    int start_offset = data.GetValues<int32_t>(1)[0];
    int end_offset = data.GetValues<int32_t>(1)[data.length + 1];
    buffer_sizes->push_back(end_offset - start_offset);
    data_buffers->push_back(data.buffers[2]->data() + start_offset);
    return Status::OK();
  }

  if (arrow::is_large_binary_like(type->id())) {
    int start_offset = (int) data.GetValues<int64_t>(1)[0];
    int end_offset = (int) data.GetValues<int64_t>(1)[data.length + 1];
    buffer_sizes->push_back(end_offset - start_offset);
    data_buffers->push_back(data.buffers[2]->data() + start_offset);
    return Status::OK();
  }

  return {Code::Invalid, "unsupported data type for serialization " + type->ToString()};
}

Status CollectOffsetBuffer(const arrow::ArrayData &data, std::vector<int32_t> *buffer_sizes,
                           std::vector<const uint8_t *> *data_buffers) {
  const auto &type = data.type;
  if (arrow::is_fixed_width(type->id())) {
    buffer_sizes->push_back(0);
    data_buffers->push_back(nullptr);
    return Status::OK();
  }

  if (arrow::is_binary_like(type->id())) {
    buffer_sizes->push_back((int) (data.length * sizeof(int32_t)));
    data_buffers->push_back(data.GetValues<uint8_t>(1));
    return Status::OK();
  }

  if (arrow::is_large_binary_like(type->id())) {
    buffer_sizes->push_back((int) (data.length * sizeof(int64_t)));
    data_buffers->push_back(data.GetValues<uint8_t>(1));
    return Status::OK();
  }

  return {Code::Invalid, "unsupported offset type for serialization " + type->ToString()};
}

Status MakeTableSerializer(const std::shared_ptr<Table> &table,
                           std::shared_ptr<TableSerializer> *serializer) {
  std::vector<int32_t> buffer_sizes;
  std::vector<const uint8_t *> data_buffers;
  // we can only send byte boundary buffers. If we encounter bitmaps that don't align to a byte
  // boundary, make a copy and keep it in this vector
  arrow::BufferVector bitmaps_with_offset;

  int num_buffers = 3 * table->Columns();
  auto atable = table->get_table();
  auto pool = ToArrowPool(table->GetContext());

  if (table->Rows() == 0) {
    buffer_sizes = std::vector<int32_t>(num_buffers, 0);
    data_buffers = std::vector<const uint8_t *>(num_buffers, nullptr);
  } else {
    // order: validity, offsets, data
    COMBINE_CHUNKS_RETURN_CYLON_STATUS(atable, pool);

    for (const auto &col: atable->columns()) {
      const auto &data = *col->chunk(0)->data();
      RETURN_CYLON_STATUS_IF_FAILED(
          CollectBitmapInfo(data, &buffer_sizes, &data_buffers, &bitmaps_with_offset, pool));
      RETURN_CYLON_STATUS_IF_FAILED(
          CollectOffsetBuffer(data, &buffer_sizes, &data_buffers));
      RETURN_CYLON_STATUS_IF_FAILED(
          CollectDataBuffer(data, &buffer_sizes, &data_buffers, &bitmaps_with_offset, pool));
    }
  }

  *serializer = std::make_shared<CylonTableSerializer>(std::move(atable), std::move(buffer_sizes),
                                                       std::move(data_buffers));
  return Status::OK();
}

int32_t CalculateNumRows(const std::shared_ptr<arrow::Schema> &schema,
                         const std::vector<int32_t> &buffer_sizes) {
  assert((int) buffer_sizes.size() == schema->num_fields() * 3);
  for (int i = 0; i < schema->num_fields(); i++) {
    const auto &type = schema->field(i)->type();
    if (type->id() == arrow::Type::BOOL) {
      continue; // bool arrays can not compute rows!
    }

    if (arrow::is_fixed_width(type->id())) {
      int byte_width =
          std::static_pointer_cast<arrow::FixedWidthType>(type)->bit_width() / CHAR_BIT;
      return buffer_sizes[3 * i + 2] / byte_width;
    }

    if (arrow::is_binary_like(type->id())) {
      return int(buffer_sizes[3 * i + 1] / sizeof(int32_t)) - 1;
    }

    if (arrow::is_large_binary_like(type->id())) {
      return int(buffer_sizes[3 * i + 1] / sizeof(int64_t)) - 1;
    }
  }

  return -1;
}

std::shared_ptr<arrow::ArrayData> MakeArrayData(std::shared_ptr<arrow::DataType> type,
                                                int64_t len,
                                                std::shared_ptr<arrow::Buffer> valid_buf,
                                                std::shared_ptr<arrow::Buffer> offset_buf,
                                                std::shared_ptr<arrow::Buffer> data_buf) {
  if (arrow::is_fixed_width(type->id())) {
    return arrow::ArrayData::Make(std::move(type),
                                  len,
                                  arrow::BufferVector{std::move(valid_buf), std::move(data_buf)});
  }

  if (arrow::is_base_binary_like(type->id())) {
    return arrow::ArrayData::Make(std::move(type),
                                  len,
                                  arrow::BufferVector{std::move(valid_buf), std::move(offset_buf),
                                                      std::move(data_buf)});
  }

  return nullptr;
}

Status DeserializeTable(const std::shared_ptr<CylonContext> &ctx,
                        const std::shared_ptr<arrow::Schema> &schema,
                        const std::vector<std::shared_ptr<cylon::Buffer>> &received_buffers,
                        const std::vector<int32_t> &disp_per_buffer,
                        const std::vector<int32_t> &buffer_sizes,
                        std::shared_ptr<Table> *output) {
  const int32_t num_cols = schema->num_fields();
  const int32_t num_rows = CalculateNumRows(schema, buffer_sizes);
  if (num_rows == -1) {
    return {Code::ExecutionError, "unable to calculate num rows for the buffers"};
  }

  arrow::ChunkedArrayVector arrays;
  arrays.reserve(num_cols);
  int i = 0, buf_cnt = 0;
  for (; i < num_cols; i++, buf_cnt += 3) {
    uint8_t *valid_start = received_buffers[buf_cnt]->GetByteBuffer() + disp_per_buffer[buf_cnt];
    uint8_t *offset_start =
        received_buffers[buf_cnt + 1]->GetByteBuffer() + disp_per_buffer[buf_cnt + 1];
    uint8_t
        *data_start = received_buffers[buf_cnt + 2]->GetByteBuffer() + disp_per_buffer[buf_cnt + 2];

    auto valid_buf =
        buffer_sizes[buf_cnt] ? std::make_shared<arrow::Buffer>(valid_start, buffer_sizes[buf_cnt])
                              : nullptr;
    auto offset_buf =
        buffer_sizes[buf_cnt + 1] ? std::make_shared<arrow::Buffer>(offset_start,
                                                                    buffer_sizes[buf_cnt + 1])
                                  : nullptr;
    auto data_buf =
        buffer_sizes[buf_cnt + 2] ? std::make_shared<arrow::Buffer>(data_start,
                                                                    buffer_sizes[buf_cnt + 2])
                                  : nullptr;

    auto data = MakeArrayData(schema->field(i)->type(), num_rows, std::move(valid_buf),
                              std::move(offset_buf), std::move(data_buf));
    arrays.push_back(std::make_shared<arrow::ChunkedArray>(arrow::MakeArray(data)));
  }

  return Table::FromArrowTable(ctx,
                               arrow::Table::Make(schema, std::move(arrays), num_rows),
                               *output);
}

}
