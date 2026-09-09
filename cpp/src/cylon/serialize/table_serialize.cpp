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

#include "cylon/arrow/arrow_buffer.hpp"
#include "cylon/arrow/arrow_types.hpp"

namespace cylon {

template<int buf_idx = 0>
Status CollectBitmapInfo(const arrow::ArrayData &data, int32_t *buffer_sizes,
                         const uint8_t **data_buffers,
                         arrow::BufferVector *bitmaps_with_offset,
                         arrow::MemoryPool *pool) {
  if (buf_idx == 0 && !data.MayHaveNulls()) {
    *buffer_sizes = 0;
    *data_buffers = nullptr;
    return Status::OK();
  }

  // there are nulls
  *buffer_sizes = (int32_t) arrow::bit_util::BytesForBits(data.length);
  if (data.offset == 0) { // no offset
    *data_buffers = data.buffers[buf_idx]->data();
  } else if (data.offset % CHAR_BIT == 0) { // offset is at a byte boundary
    *data_buffers = data.buffers[buf_idx]->data() + data.offset / CHAR_BIT;
  } else { // non-byte boundary offset
    CYLON_ASSIGN_OR_RAISE(
        auto buf,
        arrow::internal::CopyBitmap(pool, data.buffers[buf_idx]->data(), data.offset, data.length))
    *data_buffers = buf->data();
    bitmaps_with_offset->push_back(std::move(buf));
  }

  return Status::OK();
}

Status CollectDataBuffer(const arrow::ArrayData &data, int32_t *buffer_sizes,
                         const uint8_t **data_buffers,
                         arrow::BufferVector *bitmaps_with_offset,
                         arrow::MemoryPool *pool) {
  const auto &type = data.type;
  if (type->id() == arrow::Type::BOOL) {
    return CollectBitmapInfo<1>(data, buffer_sizes, data_buffers, bitmaps_with_offset, pool);
  }

  // A fixed size list keeps its values in a child array rather than in its own
  // buffers, but it needs no offsets buffer (the width is in the type), so the
  // child's values travel in the data slot and the layout stays three buffers
  // per column. Checked before is_fixed_width so the width of the *list* can
  // never be mistaken for the width of a scalar.
  if (type->id() == arrow::Type::FIXED_SIZE_LIST) {
    const auto &list_type = std::static_pointer_cast<arrow::FixedSizeListType>(type);
    const auto &value_type = list_type->value_type();
    if (!arrow::is_fixed_width(value_type->id())) {
      return {Code::Invalid,
              "unsupported fixed size list value type for serialization " + value_type->ToString()};
    }
    const int width = list_type->list_size();
    const int byte_width =
        std::static_pointer_cast<arrow::FixedWidthType>(value_type)->bit_width() / CHAR_BIT;
    const auto &child = *data.child_data[0];
    *buffer_sizes = byte_width * width * (int) data.length;
    *data_buffers = child.buffers[1]->data()
                    + (data.offset * width + child.offset) * byte_width;
    return Status::OK();
  }

  if (arrow::is_fixed_width(type->id())) {
    int byte_width = std::static_pointer_cast<arrow::FixedWidthType>(type)->bit_width() / CHAR_BIT;
    *buffer_sizes = byte_width * (int) data.length;
    *data_buffers = data.buffers[1]->data() + data.offset * byte_width;
    return Status::OK();
  }

  if (arrow::is_binary_like(type->id())) {
    int start_offset = data.GetValues<int32_t>(1)[0];
    int end_offset = data.GetValues<int32_t>(1)[data.length];
    *buffer_sizes = end_offset - start_offset;
    *data_buffers = data.buffers[2]->data() + start_offset;
    return Status::OK();
  }

  if (arrow::is_large_binary_like(type->id())) {
    int start_offset = (int) data.GetValues<int64_t>(1)[0];
    int end_offset = (int) data.GetValues<int64_t>(1)[data.length];
    *buffer_sizes = end_offset - start_offset;
    *data_buffers = data.buffers[2]->data() + start_offset;
    return Status::OK();
  }

  return {Code::Invalid, "unsupported data type for serialization " + type->ToString()};
}

Status CollectOffsetBuffer(const arrow::ArrayData &data,
                           int32_t *buffer_sizes,
                           const uint8_t **data_buffers) {
  const auto &type = data.type;
  // A fixed size list has no offsets buffer of its own — its width is carried by
  // the type — so it shares the fixed-width case's empty offset slot.
  if (arrow::is_fixed_width(type->id()) || type->id() == arrow::Type::FIXED_SIZE_LIST) {
    *buffer_sizes = 0;
    *data_buffers = nullptr;
    return Status::OK();
  }

  if (arrow::is_binary_like(type->id())) {
    *buffer_sizes = (int) ((data.length + 1) * sizeof(int32_t));
    *data_buffers = reinterpret_cast<const uint8_t *>(data.GetValues<int32_t>(1));
    return Status::OK();
  }

  if (arrow::is_large_binary_like(type->id())) {
    *buffer_sizes = (int) ((data.length + 1) * sizeof(int64_t));
    *data_buffers = reinterpret_cast<const uint8_t *>(data.GetValues<int64_t>(1));
    return Status::OK();
  }

  return {Code::Invalid, "unsupported offset type for serialization " + type->ToString()};
}


// Buffer slots a field occupies on the wire. Three for the node itself
// (validity, offsets, data) plus its children's, so every type that existed
// before nesting was supported still occupies exactly three and the layout is
// unchanged for them. A fixed size list is deliberately not recursed into: it
// carries its child's values in its own data slot (see CollectDataBuffer).
// Both ends compute this from the same Arrow schema, so no count travels on
// the wire.
int32_t BufferSlots(const std::shared_ptr<arrow::DataType> &type) {
  if (type->id() == arrow::Type::FIXED_SIZE_LIST) {
    return 3;
  }
  int32_t slots = 3;
  for (int i = 0; i < type->num_fields(); i++) {
    slots += BufferSlots(type->field(i)->type());
  }
  return slots;
}

int32_t BufferSlots(const std::shared_ptr<arrow::Schema> &schema) {
  int32_t slots = 0;
  for (const auto &field : schema->fields()) {
    slots += BufferSlots(field->type());
  }
  return slots;
}

Status CollectArrayBuffers(const arrow::ArrayData &data, int32_t *buffer_sizes,
                           const uint8_t **data_buffers,
                           arrow::BufferVector *bitmaps_with_offset,
                           arrow::MemoryPool *pool) {
  RETURN_CYLON_STATUS_IF_FAILED(
      CollectBitmapInfo(data, &buffer_sizes[0], &data_buffers[0], bitmaps_with_offset, pool));

  if (data.type->id() == arrow::Type::STRUCT) {
    // A struct holds nothing of its own beyond the validity bitmap; its values
    // live entirely in the children, each of which claims its own slots.
    if (data.offset != 0) {
      return {Code::Invalid, "cannot serialize a sliced struct array"};
    }
    buffer_sizes[1] = 0;
    data_buffers[1] = nullptr;
    buffer_sizes[2] = 0;
    data_buffers[2] = nullptr;

    int32_t at = 3;
    for (const auto &child : data.child_data) {
      RETURN_CYLON_STATUS_IF_FAILED(
          CollectArrayBuffers(*child, &buffer_sizes[at], &data_buffers[at], bitmaps_with_offset,
                              pool));
      at += BufferSlots(child->type);
    }
    return Status::OK();
  }

  RETURN_CYLON_STATUS_IF_FAILED(
      CollectOffsetBuffer(data, &buffer_sizes[1], &data_buffers[1]));
  return CollectDataBuffer(data, &buffer_sizes[2], &data_buffers[2], bitmaps_with_offset, pool);
}

Status CylonTableSerializer::Make(const std::shared_ptr<Table> &table,
                                  std::shared_ptr<TableSerializer> *serializer) {
  // we can only send byte boundary buffers. If we encounter bitmaps that don't align to a byte
  // boundary, make a copy and keep it in this vector
  arrow::BufferVector extra_buffers;

  auto atable = table->get_table();
  auto pool = ToArrowPool(table->GetContext());
  int num_buffers = BufferSlots(atable->schema());

  std::vector<int32_t> buffer_sizes(num_buffers, 0);
  std::vector<const uint8_t *> data_buffers(num_buffers, nullptr);

  if (table->Rows()) {
    // order per node: validity, offsets, data — then each child's slots
    COMBINE_CHUNKS_RETURN_CYLON_STATUS(atable, pool);

    int32_t at = 0;
    for (int i = 0; i < atable->num_columns(); i++) {
      const auto &data = *atable->column(i)->chunk(0)->data();
      RETURN_CYLON_STATUS_IF_FAILED(
          CollectArrayBuffers(data, &buffer_sizes[at], &data_buffers[at], &extra_buffers, pool));
      at += BufferSlots(data.type);
    }
  }

  *serializer = std::make_shared<CylonTableSerializer>(std::move(atable),
                                                       std::move(buffer_sizes),
                                                       std::move(data_buffers),
                                                       std::move(extra_buffers));
  return Status::OK();
}

CylonTableSerializer::CylonTableSerializer(std::shared_ptr<arrow::Table> table,
                                           std::vector<int32_t> buffer_sizes,
                                           std::vector<const uint8_t *> data_buffers,
                                           arrow::BufferVector extra_buffers)
    : table_(std::move(table)),
      num_buffers_(BufferSlots(table_->schema())),
      buffer_sizes_(std::move(buffer_sizes)),
      data_buffers_(std::move(data_buffers)),
      extra_buffers_(std::move(extra_buffers)) {
  assert(num_buffers_ == (int) buffer_sizes_.size());
  assert(num_buffers_ == (int) data_buffers_.size());
}

std::vector<int32_t> CylonTableSerializer::getDataTypes() {

  std::vector<int32_t> data_types;
  data_types.reserve(table_->num_columns());
  for (const auto &f: table_->schema()->fields()) {
    data_types.push_back(static_cast<int32_t>(tarrow::ToCylonTypeId(f->type())));
  }
  return data_types;
}

std::vector<int32_t> CylonTableSerializer::getEmptyTableBufferSizes() {
  return std::vector<int32_t>(num_buffers_, 0);
}

const arrow::BufferVector &CylonTableSerializer::extra_buffers() const {
  return extra_buffers_;
}

int32_t CalculateNumRows(const std::shared_ptr<arrow::DataType> &type,
                         const std::array<int32_t, 3> &buffer_sizes) {
  if (type->id() == arrow::Type::BOOL) {
    return -1; // bool arrays can not compute rows!
  }

  if (type->id() == arrow::Type::FIXED_SIZE_LIST) {
    const auto &list_type = std::static_pointer_cast<arrow::FixedSizeListType>(type);
    const int byte_width =
        std::static_pointer_cast<arrow::FixedWidthType>(list_type->value_type())->bit_width()
            / CHAR_BIT;
    return buffer_sizes[2] / (byte_width * list_type->list_size());
  }

  if (arrow::is_fixed_width(type->id())) {
    int byte_width =
        std::static_pointer_cast<arrow::FixedWidthType>(type)->bit_width() / CHAR_BIT;
    return buffer_sizes[2] / byte_width;
  }

  // An empty contribution carries no offsets at all, not the single sentinel
  // offset a 0-row array would normally hold, so size 0 means 0 rows rather
  // than an uncomputable length. Returning -1 here made a rank that
  // contributed nothing look undecodable, and its table was dropped from the
  // gather entirely.
  if (arrow::is_binary_like(type->id())) {
    return buffer_sizes[1] ? int(buffer_sizes[1] / sizeof(int32_t)) - 1 : 0;
  }

  if (arrow::is_large_binary_like(type->id())) {
    return buffer_sizes[1] ? int(buffer_sizes[1] / sizeof(int64_t)) - 1 : 0;
  }

  return -1;
}

int32_t CalculateNumRows(const std::shared_ptr<arrow::Schema> &schema,
                         const std::vector<int32_t> &buffer_sizes) {
  assert((int) buffer_sizes.size() == BufferSlots(schema));
  int slot = 0;
  for (int i = 0; i < schema->num_fields(); i++) {
    const auto &type = schema->field(i)->type();
    const int base = slot;
    slot += BufferSlots(type);
    // A struct carries no length of its own here; its first child does, and the
    // child's slots follow immediately after the struct's three.
    if (type->id() == arrow::Type::STRUCT) {
      const auto &child_type = type->field(0)->type();
      std::array<int32_t, 3> child{buffer_sizes[base + 3], buffer_sizes[base + 4],
                                   buffer_sizes[base + 5]};
      int32_t rows = CalculateNumRows(child_type, child);
      if (rows != -1) {
        return rows;
      }
      continue;
    }
    if (type->id() == arrow::Type::BOOL) {
      continue; // bool arrays can not compute rows!
    }

    if (type->id() == arrow::Type::FIXED_SIZE_LIST) {
      const auto &list_type = std::static_pointer_cast<arrow::FixedSizeListType>(type);
      const int byte_width =
          std::static_pointer_cast<arrow::FixedWidthType>(list_type->value_type())->bit_width()
              / CHAR_BIT;
      return buffer_sizes[base + 2] / (byte_width * list_type->list_size());
    }

    if (arrow::is_fixed_width(type->id())) {
      int byte_width =
          std::static_pointer_cast<arrow::FixedWidthType>(type)->bit_width() / CHAR_BIT;
      return buffer_sizes[base + 2] / byte_width;
    }

    if (arrow::is_binary_like(type->id())) {
      return buffer_sizes[base + 1] ? int(buffer_sizes[base + 1] / sizeof(int32_t)) - 1 : 0;
    }

    if (arrow::is_large_binary_like(type->id())) {
      return buffer_sizes[base + 1] ? int(buffer_sizes[base + 1] / sizeof(int64_t)) - 1 : 0;
    }
  }

  return -1;
}

template<typename OffsetType>
std::shared_ptr<arrow::ArrayData> MakeArrayDataBinary(std::shared_ptr<arrow::DataType> type,
                                                      int64_t len,
                                                      std::shared_ptr<arrow::Buffer> valid_buf,
                                                      std::shared_ptr<arrow::Buffer> offset_buf,
                                                      std::shared_ptr<arrow::Buffer> data_buf) {
  if (len) { // only do this for non-empty arrays
    auto *offsets = reinterpret_cast<OffsetType *>(offset_buf->mutable_data());
    OffsetType start_offset = offsets[0];

    if (start_offset) { // if there's a start offset, remove it
      for (int64_t i = 0; i < len + 1; i++) {
        offsets[i] -= start_offset;
      }
    }
  }
  return arrow::ArrayData::Make(std::move(type), len, {std::move(valid_buf), std::move(offset_buf),
                                                       std::move(data_buf)});
}

std::shared_ptr<arrow::ArrayData> MakeArrayData(std::shared_ptr<arrow::DataType> type,
                                                int64_t len,
                                                std::shared_ptr<arrow::Buffer> valid_buf,
                                                std::shared_ptr<arrow::Buffer> offset_buf,
                                                std::shared_ptr<arrow::Buffer> data_buf) {
  // Rebuild the child array the values travelled in; the parent carries only
  // its validity bitmap, exactly mirroring how CollectDataBuffer took it apart.
  if (type->id() == arrow::Type::FIXED_SIZE_LIST) {
    const auto &list_type = std::static_pointer_cast<arrow::FixedSizeListType>(type);
    auto child = arrow::ArrayData::Make(list_type->value_type(),
                                        len * list_type->list_size(),
                                        {nullptr, std::move(data_buf)});
    return arrow::ArrayData::Make(std::move(type), len, {std::move(valid_buf)},
                                  {std::move(child)});
  }

  if (arrow::is_fixed_width(type->id())) {
    return arrow::ArrayData::Make(std::move(type), len, {std::move(valid_buf),
                                                         std::move(data_buf)});
  }

  if (arrow::is_binary_like(type->id())) {
    return MakeArrayDataBinary<int32_t>(std::move(type), len, std::move(valid_buf),
                                        std::move(offset_buf), std::move(data_buf));
  }

  if (arrow::is_large_binary_like(type->id())) {
    return MakeArrayDataBinary<int64_t>(std::move(type), len, std::move(valid_buf),
                                        std::move(offset_buf), std::move(data_buf));
  }

  return nullptr;
}

std::shared_ptr<arrow::Buffer> MakeArrowBuffer(const std::shared_ptr<Buffer> &buffer,
                                               int32_t offset, int32_t size) {
  const auto &arrow_buf = std::static_pointer_cast<ArrowBuffer>(buffer)->getBuf();
  // we need to slice the buffer here, rather than creating an arrow::Buffer from cylon::buffer data
  // pointer. Because we need to pass the ownership of the parent buffer to the slice. otherwise,
  // the buffers would have to be managed at the allocator level.
  return SliceMutableBuffer(arrow_buf, offset, size);
}


// Rebuild one field from its slots, mirroring CollectArrayBuffers. Returns how
// many slots it consumed so the caller can advance to the next field.
std::shared_ptr<arrow::ArrayData> MakeArrayDataRec(
    const std::shared_ptr<arrow::DataType> &type, int64_t len,
    const std::vector<std::shared_ptr<Buffer>> &received_buffers,
    const std::vector<int32_t> &buffer_sizes,
    const std::vector<int32_t> &buffer_offsets,
    int base) {
  auto slot = [&](int k) -> std::shared_ptr<arrow::Buffer> {
    return buffer_sizes[base + k] ? MakeArrowBuffer(received_buffers[base + k],
                                                    buffer_offsets[base + k],
                                                    buffer_sizes[base + k])
                                  : nullptr;
  };

  if (type->id() == arrow::Type::STRUCT) {
    std::vector<std::shared_ptr<arrow::ArrayData>> children;
    int at = base + 3;
    for (int i = 0; i < type->num_fields(); i++) {
      const auto &child_type = type->field(i)->type();
      children.push_back(MakeArrayDataRec(child_type, len, received_buffers, buffer_sizes,
                                          buffer_offsets, at));
      at += BufferSlots(child_type);
    }
    return arrow::ArrayData::Make(type, len, {slot(0)}, std::move(children));
  }

  auto data_buf = MakeArrowBuffer(received_buffers[base + 2], buffer_offsets[base + 2],
                                  buffer_sizes[base + 2]);
  return MakeArrayData(type, len, slot(0), slot(1), std::move(data_buf));
}

Status DeserializeTable(const std::shared_ptr<CylonContext> &ctx,
                        const std::shared_ptr<arrow::Schema> &schema,
                        const std::vector<std::shared_ptr<Buffer>> &received_buffers,
                        const std::vector<int32_t> &buffer_sizes,
                        const std::vector<int32_t> &buffer_offsets,
                        std::shared_ptr<Table> *output,
                        int32_t num_rows) {
  assert(received_buffers.size() == buffer_sizes.size());
  assert(received_buffers.size() == buffer_offsets.size());
  assert(received_buffers.size() == (size_t) BufferSlots(schema));

  const int32_t num_cols = schema->num_fields();
  const int32_t rows = num_rows >= 0 ? num_rows : CalculateNumRows(schema, buffer_sizes);
  if (rows == -1) {
    return {Code::ExecutionError, "unable to calculate num rows for the buffers"};
  }

  arrow::ChunkedArrayVector arrays;
  arrays.reserve(num_cols);
  for (int i = 0, b = 0; i < num_cols; i++) {
    const auto &type = schema->field(i)->type();
    auto data = MakeArrayDataRec(type, rows, received_buffers, buffer_sizes, buffer_offsets, b);
    b += BufferSlots(type);
    arrays.push_back(std::make_shared<arrow::ChunkedArray>(arrow::MakeArray(data)));
  }

  return Table::FromArrowTable(ctx,
                               arrow::Table::Make(schema, std::move(arrays), rows),
                               *output);
}

Status DeserializeTable(const std::shared_ptr<CylonContext> &ctx,
                        const std::shared_ptr<arrow::Schema> &schema,
                        const std::vector<std::shared_ptr<Buffer>> &received_buffers,
                        const std::vector<int32_t> &buffer_sizes,
                        std::shared_ptr<Table> *output,
                        int32_t num_rows) {
  assert(received_buffers.size() == buffer_sizes.size());
  assert(received_buffers.size() == (size_t) BufferSlots(schema));

  std::vector<int32_t> buffer_offsets(buffer_sizes.size(), 0);
  return DeserializeTable(ctx, schema, received_buffers, buffer_sizes, buffer_offsets, output,
                          num_rows);
}

Status DeserializeTable(const std::shared_ptr<CylonContext> &ctx,
                        const std::shared_ptr<arrow::Schema> &schema,
                        const std::vector<std::shared_ptr<Buffer>> &received_buffers,
                        std::shared_ptr<Table> *output,
                        int32_t num_rows) {
  std::vector<int32_t> buffer_sizes(received_buffers.size());
  for (size_t i = 0; i < received_buffers.size(); i++) {
    buffer_sizes[i] = (int32_t) received_buffers[i]->GetLength();
  }
  return DeserializeTable(ctx, schema, received_buffers, buffer_sizes, output, num_rows);
}

Status DeserializeTables(const std::shared_ptr<CylonContext> &ctx,
                         const std::shared_ptr<arrow::Schema> &schema,
                         int num_tables,
                         const std::vector<std::shared_ptr<Buffer>> &received_buffers,
                         const std::vector<int32_t> &buffer_sizes_per_table,
                         const std::vector<int32_t> &buffer_offsets_per_table,
                         std::vector<std::shared_ptr<Table>> *output,
                         const std::vector<int32_t> &num_rows_per_table) {
  const int num_buffers = BufferSlots(schema);

  assert((int) received_buffers.size() == num_buffers);
  assert((int) buffer_sizes_per_table.size() == num_tables * num_buffers);
  assert((int) buffer_offsets_per_table.size() == num_tables * num_buffers);
  assert(num_rows_per_table.empty() || (int) num_rows_per_table.size() == num_tables);

  output->reserve(num_tables);
  std::vector<int32_t> table_buffer_sizes(num_buffers, 0), table_buffer_offsets(num_buffers, 0);
  for (int i = 0; i < num_tables; i++) {
    std::shared_ptr<Table> out;
    int offset = i * num_buffers;
    std::copy(buffer_sizes_per_table.begin() + offset,
              buffer_sizes_per_table.begin() + offset + num_buffers,
              table_buffer_sizes.begin());

    std::copy(buffer_offsets_per_table.begin() + offset,
              buffer_offsets_per_table.begin() + offset + num_buffers,
              table_buffer_offsets.begin());
    RETURN_CYLON_STATUS_IF_FAILED(DeserializeTable(ctx, schema, received_buffers,
                                                   table_buffer_sizes, table_buffer_offsets, &out,
                                                   num_rows_per_table.empty()
                                                       ? -1 : num_rows_per_table[i]));
    output->push_back(std::move(out));
  }

  return Status::OK();
}

CylonColumnSerializer::CylonColumnSerializer(std::shared_ptr<arrow::Array> array,
                                             const std::array<const uint8_t *, 3> &data_bufs,
                                             const std::array<int32_t, 3> &buf_sizes,
                                             arrow::BufferVector extra_buffers)
    : array_(std::move(array)),
      data_bufs_(data_bufs),
      buf_sizes_(buf_sizes),
      extra_buffers_(std::move(extra_buffers)) {}

const std::array<const uint8_t *, 3> &CylonColumnSerializer::data_buffers() const {
  return data_bufs_;
}

const std::array<int32_t, 3> &CylonColumnSerializer::buffer_sizes() const {
  return buf_sizes_;
}

int32_t CylonColumnSerializer::getDataTypeId() const {
  return static_cast<int32_t>(tarrow::ToCylonTypeId(array_->type()));
}

Status CylonColumnSerializer::Make(const std::shared_ptr<arrow::Array> &column,
                                   std::shared_ptr<ColumnSerializer> *serializer,
                                   arrow::MemoryPool *pool) {
  std::array<int32_t, 3> buffer_sizes{};
  std::array<const uint8_t *, 3> data_buffers{};
  // we can only send byte boundary buffers. If we encounter bitmaps that don't align to a byte
  // boundary, make a copy and keep it in this vector
  arrow::BufferVector extra_buffers;

  if (column->length()) {
    // order: validity, offsets, data
    const auto &data = *column->data();
    RETURN_CYLON_STATUS_IF_FAILED(
        CollectBitmapInfo(data, &buffer_sizes[0], &data_buffers[0], &extra_buffers, pool));
    RETURN_CYLON_STATUS_IF_FAILED(
        CollectOffsetBuffer(data, &buffer_sizes[1], &data_buffers[1]));
    RETURN_CYLON_STATUS_IF_FAILED(
        CollectDataBuffer(data, &buffer_sizes[2], &data_buffers[2], &extra_buffers, pool));
  }

  *serializer = std::make_shared<CylonColumnSerializer>(column, data_buffers, buffer_sizes,
                                                        std::move(extra_buffers));
  return Status::OK();
}

Status CylonColumnSerializer::Make(const std::shared_ptr<arrow::ChunkedArray> &column,
                                   std::shared_ptr<ColumnSerializer> *serializer,
                                   arrow::MemoryPool *pool) {
  if (column->num_chunks() == 1) {
    return Make(column->chunk(0), serializer, pool);
  }
  CYLON_ASSIGN_OR_RAISE(auto arr, arrow::Concatenate(column->chunks(), pool))
  return Make(arr, serializer, pool);
}

Status CylonColumnSerializer::Make(const std::shared_ptr<Column> &column,
                                   std::shared_ptr<ColumnSerializer> *serializer,
                                   MemoryPool *pool) {
  return Make(column->data(), serializer, ToArrowPool(pool));
}

Status DeserializeColumn(const std::shared_ptr<arrow::DataType> &data_type,
                         const std::array<std::shared_ptr<Buffer>, 3> &received_buffers,
                         const std::array<int32_t, 3> &buffer_sizes,
                         const std::array<int32_t, 3> &buffer_offsets,
                         std::shared_ptr<Column> *output) {
  if (data_type->id() == arrow::Type::BOOL) {
    return {Code::Invalid, "deserializing bool type column is not supported"};
  }

  int num_rows = CalculateNumRows(data_type, buffer_sizes);
  if (num_rows == -1) {
    return {Code::ExecutionError, "unable to calculate num rows for the buffers"};
  }

  auto valid_buf = buffer_sizes[0] ? MakeArrowBuffer(received_buffers[0], buffer_offsets[0],
                                                     buffer_sizes[0])
                                   : nullptr;
  auto offset_buf = buffer_sizes[1] ? MakeArrowBuffer(received_buffers[1], buffer_offsets[1],
                                                      buffer_sizes[1])
                                    : nullptr;
  auto data_buf = MakeArrowBuffer(received_buffers[2], buffer_offsets[2],
                                  buffer_sizes[2]);

  auto data = MakeArrayData(data_type, num_rows, std::move(valid_buf),
                            std::move(offset_buf), std::move(data_buf));

  *output = Column::Make(arrow::MakeArray(data));
  return Status::OK();
}

Status DeserializeColumn(const std::shared_ptr<arrow::DataType> &data_type,
                         const std::array<std::shared_ptr<Buffer>, 3> &received_buffers,
                         const std::array<int32_t, 3> &buffer_sizes,
                         std::shared_ptr<Column> *output) {
  return DeserializeColumn(data_type, received_buffers, buffer_sizes, std::array<int32_t, 3>{},
                           output);
}

}
