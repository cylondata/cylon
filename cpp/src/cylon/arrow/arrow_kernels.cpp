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

#include "arrow_kernels.hpp"

#include <vector>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <memory>

namespace cylon {
std::unique_ptr<ArrowArraySplitKernel> CreateSplitter(const std::shared_ptr<arrow::DataType> &type,
                                                      arrow::MemoryPool *pool) {
  switch (type->id()) {
    case arrow::Type::UINT8:return std::make_unique<UInt8ArraySplitter>(pool);
    case arrow::Type::INT8:return std::make_unique<Int8ArraySplitter>(pool);
    case arrow::Type::UINT16:return std::make_unique<UInt16ArraySplitter>(pool);
    case arrow::Type::INT16:return std::make_unique<Int16ArraySplitter>(pool);
    case arrow::Type::UINT32:return std::make_unique<UInt32ArraySplitter>(pool);
    case arrow::Type::INT32:return std::make_unique<Int32ArraySplitter>(pool);
    case arrow::Type::UINT64:return std::make_unique<UInt64ArraySplitter>(pool);
    case arrow::Type::INT64:return std::make_unique<Int64ArraySplitter>(pool);
    case arrow::Type::FLOAT:return std::make_unique<FloatArraySplitter>(pool);
    case arrow::Type::DOUBLE:return std::make_unique<DoubleArraySplitter>(pool);
    case arrow::Type::FIXED_SIZE_BINARY:return std::make_unique<FixedBinaryArraySplitKernel>(pool);
    case arrow::Type::STRING:return std::make_unique<BinaryArraySplitKernel<arrow::StringType>>(pool);
    case arrow::Type::BINARY:return std::make_unique<BinaryArraySplitKernel<arrow::BinaryType>>(pool);
    default: return nullptr;
  }
}

Status FixedBinaryArraySplitKernel::Split(const std::shared_ptr<arrow::ChunkedArray> &values,
                                          uint32_t num_partitions,
                                          const std::vector<uint32_t> &target_partitions,
                                          const std::vector<uint32_t> &counts,
                                          std::vector<std::shared_ptr<arrow::Array>> &output) {
  if ((size_t) values->length() != target_partitions.size()) {
    return Status(Code::ExecutionError, "values rows != target_partitions length");
  }

  std::vector<std::unique_ptr<ARROW_BUILDER_T>> builders;
  builders.reserve(num_partitions);
  for (uint32_t i = 0; i < num_partitions; i++) {
    builders.emplace_back(new ARROW_BUILDER_T(values->type(), pool_));
    const auto &status = builders.back()->Reserve(counts[i]);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(status)
  }

  size_t offset = 0;
  for (const auto &array:values->chunks()) {
    std::shared_ptr<ARROW_ARRAY_T> casted_array = std::static_pointer_cast<ARROW_ARRAY_T>(array);
    const int64_t arr_len = array->length();
    for (int64_t i = 0; i < arr_len; i++, offset++) {
      const auto &a_status = builders[target_partitions[offset]]->Append(casted_array->Value(i));
      RETURN_CYLON_STATUS_IF_ARROW_FAILED(a_status)
    }
  }

  output.reserve(num_partitions);
  for (uint32_t i = 0; i < num_partitions; i++) {
    std::shared_ptr<arrow::Array> array;
    const auto &status = builders[i]->Finish(&array);
    RETURN_CYLON_STATUS_IF_ARROW_FAILED(status)
    output.push_back(array);
  }

  return Status::OK();
}


class ArrowStringSortKernel : public IndexSortKernel {
 public:
  explicit ArrowStringSortKernel(arrow::MemoryPool *pool) : IndexSortKernel(pool) {}

  arrow::Status Sort(std::shared_ptr<arrow::Array> &values, std::shared_ptr<arrow::Array> &offsets) override {
    auto array = std::static_pointer_cast<arrow::StringArray>(values);
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(int64_t);

    arrow::Result<std::unique_ptr<arrow::Buffer>> result = AllocateBuffer(buf_size + 1, pool_);
    const arrow::Status &status = result.status();
    if (!status.ok()) {
      LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return status;
    }
    indices_buf = std::move(result.ValueOrDie());

    auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
    for (int64_t i = 0; i < values->length(); i++) {
      indices_begin[i] = i;
    }

    int64_t *indices_end = indices_begin + values->length();
    std::sort(indices_begin, indices_end, [array](int64_t left, int64_t right) {
      return array->GetView(left).compare(array->GetView(right)) < 0;
    });
    offsets = std::make_shared<arrow::UInt64Array>(values->length(), indices_buf);
    return arrow::Status::OK();
  }
};

class ArrowFixedSizeBinarySortKernel : public IndexSortKernel {
 public:
  explicit ArrowFixedSizeBinarySortKernel(arrow::MemoryPool *pool) : IndexSortKernel(pool) {}

  arrow::Status Sort(std::shared_ptr<arrow::Array> &values, std::shared_ptr<arrow::Array> &offsets) override {
    auto array = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(uint64_t);
    arrow::Result<std::unique_ptr<arrow::Buffer>> result = AllocateBuffer(buf_size + 1, pool_);
    const arrow::Status &status = result.status();
    if (!status.ok()) {
      LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return status;
    }
    indices_buf = std::move(result.ValueOrDie());
    auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
    for (int64_t i = 0; i < values->length(); i++) {
      indices_begin[i] = i;
    }
    int64_t *indices_end = indices_begin + values->length();
    std::sort(indices_begin, indices_end, [array](uint64_t left, uint64_t right) {
      return array->GetView(left).compare(array->GetView(right)) < 0;
    });
    offsets = std::make_shared<arrow::UInt64Array>(values->length(), indices_buf);
    return arrow::Status::OK();
  }
};

class ArrowBinarySortKernel : public IndexSortKernel {
 public:
  explicit ArrowBinarySortKernel(arrow::MemoryPool *pool) :
      IndexSortKernel(pool) {}

  arrow::Status Sort(std::shared_ptr<arrow::Array> &values, std::shared_ptr<arrow::Array> &offsets) override {
    auto array = std::static_pointer_cast<arrow::BinaryArray>(values);
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(uint64_t);
    arrow::Result<std::unique_ptr<arrow::Buffer>> result = AllocateBuffer(buf_size + 1, pool_);
    const arrow::Status &status = result.status();
    if (!status.ok()) {
      LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return status;
    }
    indices_buf = std::move(result.ValueOrDie());
    auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
    for (int64_t i = 0; i < values->length(); i++) {
      indices_begin[i] = i;
    }
    int64_t *indices_end = indices_begin + values->length();
    std::sort(indices_begin, indices_end, [array](uint64_t left, uint64_t right) {
      return array->GetView(left).compare(array->GetView(right)) < 0;
    });
    offsets = std::make_shared<arrow::UInt64Array>(values->length(), indices_buf);
    return arrow::Status::OK();
  }
};

std::unique_ptr<IndexSortKernel> CreateSorter(const std::shared_ptr<arrow::DataType> &type,
                                              arrow::MemoryPool *pool) {
  switch (type->id()) {
    case arrow::Type::UINT8:return std::make_unique<UInt8ArraySorter>(pool);
    case arrow::Type::INT8:return std::make_unique<Int8ArraySorter>(pool);
    case arrow::Type::UINT16:return std::make_unique<UInt16ArraySorter>(pool);
    case arrow::Type::INT16:return std::make_unique<Int16ArraySorter>(pool);
    case arrow::Type::UINT32:return std::make_unique<UInt32ArraySorter>(pool);
    case arrow::Type::INT32:return std::make_unique<Int32ArraySorter>(pool);
    case arrow::Type::UINT64:return std::make_unique<UInt64ArraySorter>(pool);
    case arrow::Type::INT64:return std::make_unique<Int64ArraySorter>(pool);
    case arrow::Type::FLOAT:return std::make_unique<FloatArraySorter>(pool);
    case arrow::Type::DOUBLE:return std::make_unique<DoubleArraySorter>(pool);
    case arrow::Type::STRING:return std::make_unique<ArrowStringSortKernel>(pool);
    case arrow::Type::BINARY:return std::make_unique<ArrowBinarySortKernel>(pool);
    case arrow::Type::FIXED_SIZE_BINARY:return std::make_unique<ArrowFixedSizeBinarySortKernel>(pool);
    default: return nullptr;
  }
}

arrow::Status SortIndices(arrow::MemoryPool *memory_pool, std::shared_ptr<arrow::Array> &values,
                          std::shared_ptr<arrow::Array> &offsets) {
  std::unique_ptr<IndexSortKernel> out = CreateSorter(values->type(), memory_pool);
  if (out == nullptr) {
    return arrow::Status(arrow::StatusCode::NotImplemented, "unknown type " + values->type()->ToString());
  }
  return out->Sort(values, offsets);
}

std::unique_ptr<InplaceIndexSortKernel> CreateInplaceSorter(const std::shared_ptr<arrow::DataType> &type,
                                                            arrow::MemoryPool *pool) {
  switch (type->id()) {
    case arrow::Type::UINT8:return std::make_unique<UInt8ArrayInplaceSorter>(pool);
    case arrow::Type::INT8:return std::make_unique<Int8ArrayInplaceSorter>(pool);
    case arrow::Type::UINT16:return std::make_unique<UInt16ArrayInplaceSorter>(pool);
    case arrow::Type::INT16:return std::make_unique<Int16ArrayInplaceSorter>(pool);
    case arrow::Type::UINT32:return std::make_unique<UInt32ArrayInplaceSorter>(pool);
    case arrow::Type::INT32:return std::make_unique<Int32ArrayInplaceSorter>(pool);
    case arrow::Type::UINT64:return std::make_unique<UInt64ArrayInplaceSorter>(pool);
    case arrow::Type::INT64:return std::make_unique<Int64ArrayInplaceSorter>(pool);
    case arrow::Type::FLOAT:return std::make_unique<FloatArrayInplaceSorter>(pool);
    case arrow::Type::DOUBLE:return std::make_unique<DoubleArrayInplaceSorter>(pool);
    default: return nullptr;
  }
}

arrow::Status SortIndicesInPlace(arrow::MemoryPool *memory_pool,
                                 std::shared_ptr<arrow::Array> &values,
                                 std::shared_ptr<arrow::UInt64Array> &offsets) {
  std::unique_ptr<InplaceIndexSortKernel> out = CreateInplaceSorter(values->type(), memory_pool);
  if (out == nullptr) {
    return arrow::Status(arrow::StatusCode::NotImplemented, "unknown type " + values->type()->ToString());
  }
  return out->Sort(values, offsets);
}

}  // namespace cylon
