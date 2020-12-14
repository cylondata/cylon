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

cylon::Status CreateStreamingSplitter(const std::shared_ptr<arrow::DataType> &type,
                                      const std::vector<int32_t> &targets,
                             arrow::MemoryPool *pool,
                             std::shared_ptr<ArrowArrayStreamingSplitKernel> *out) {
  ArrowArrayStreamingSplitKernel *kernel;
  switch (type->id()) {
    case arrow::Type::UINT8:kernel = new UInt8ArrayStreamingSplitter(type, targets, pool);
      break;
    case arrow::Type::INT8:kernel = new Int8ArrayStreamingSplitter(type, targets, pool);
      break;
    case arrow::Type::UINT16:kernel = new UInt16ArrayStreamingSplitter(type, targets, pool);
      break;
    case arrow::Type::INT16:kernel = new Int16ArrayStreamingSplitter(type, targets, pool);
      break;
    case arrow::Type::UINT32:kernel = new UInt32ArrayStreamingSplitter(type, targets, pool);
      break;
    case arrow::Type::INT32:kernel = new Int32ArrayStreamingSplitter(type, targets, pool);
      break;
    case arrow::Type::UINT64:kernel = new UInt64ArrayStreamingSplitter(type, targets, pool);
      break;
    case arrow::Type::INT64:kernel = new Int64ArrayStreamingSplitter(type, targets, pool);
      break;
    case arrow::Type::FLOAT:kernel = new FloatArrayStreamingSplitter(type, targets, pool);
      break;
    case arrow::Type::DOUBLE:kernel = new DoubleArrayStreamingSplitter(type, targets, pool);
      break;
    case arrow::Type::FIXED_SIZE_BINARY:kernel =
        new FixedBinaryArrayStreamingSplitKernel(type, targets, pool);
      break;
    case arrow::Type::STRING:kernel = new BinaryArrayStreamingSplitKernel(type, targets, pool);
      break;
    case arrow::Type::BINARY:kernel = new BinaryArrayStreamingSplitKernel(type, targets, pool);
      break;
    default:
      LOG(FATAL) << "Un-known type " << type->name();
      return cylon::Status(cylon::NotImplemented, "This type not implemented");
  }
  out->reset(kernel);
  return cylon::Status::OK();
}

int FixedBinaryArrayStreamingSplitKernel::Split(std::shared_ptr<arrow::Array> &values,
                                                const std::vector<int64_t> &partitions,
                                                const std::vector<uint32_t> &cnts) {
  auto reader =
      std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
  for (size_t i = 0; i < partitions.size(); i++) {
    std::shared_ptr<arrow::FixedSizeBinaryBuilder> b = builders_[partitions.at(i)];
    if (b->Append(reader->Value(i)) != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to merge";
      return -1;
    }
  }
  return 0;
}

int FixedBinaryArrayStreamingSplitKernel::finish(
    std::unordered_map<int, std::shared_ptr<arrow::Array>> &out) {
  for (int it : targets_) {
    std::shared_ptr<arrow::FixedSizeBinaryBuilder> b = builders_[it];
    std::shared_ptr<arrow::Array> array;
    if (b->Finish(&array) != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to merge";
      return -1;
    }
    out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(it, array));
  }
  return 0;
}

FixedBinaryArrayStreamingSplitKernel::FixedBinaryArrayStreamingSplitKernel(
    const std::shared_ptr<arrow::DataType>& type,
    const std::vector<int32_t> &targets,
    arrow::MemoryPool *pool) : ArrowArrayStreamingSplitKernel(type, targets, pool) {
  for (size_t i = 0; i < targets.size(); i++) {
    int target = targets[i];
    std::shared_ptr<arrow::FixedSizeBinaryBuilder> b =
        std::make_shared<arrow::FixedSizeBinaryBuilder>(type_, pool_);
    builders_.insert(std::pair<int, std::shared_ptr<arrow::FixedSizeBinaryBuilder>>(target, b));
  }
}

int BinaryArrayStreamingSplitKernel::Split(std::shared_ptr<arrow::Array> &values,
                                           const std::vector<int64_t> &partitions,
                                           const std::vector<uint32_t> &cnts) {
  auto reader =
      std::static_pointer_cast<arrow::BinaryArray>(values);

  for (size_t i = 0; i < partitions.size(); i++) {
    std::shared_ptr<arrow::BinaryBuilder> b = builders_[partitions.at(i)];
    int length = 0;
    const uint8_t *value = reader->GetValue(i, &length);
    if (b->Append(value, length) != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to merge";
      return -1;
    }
  }
  return 0;
}

int BinaryArrayStreamingSplitKernel::finish(
    std::unordered_map<int, std::shared_ptr<arrow::Array>> &out) {
  for (int it : targets_) {
    std::shared_ptr<arrow::BinaryBuilder> b = builders_[it];
    std::shared_ptr<arrow::Array> array;
    if (b->Finish(&array) != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to merge";
      return -1;
    }
    out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(it, array));
  }
  return 0;
}

BinaryArrayStreamingSplitKernel::BinaryArrayStreamingSplitKernel(
    const std::shared_ptr<arrow::DataType>& type,
    const std::vector<int32_t> &targets,
    arrow::MemoryPool *pool) :
    ArrowArrayStreamingSplitKernel(type, targets, pool) {
  for (int it : targets_) {
    std::shared_ptr<arrow::BinaryBuilder> b = std::make_shared<arrow::BinaryBuilder>(type_, pool_);
    builders_.insert(std::pair<int, std::shared_ptr<arrow::BinaryBuilder>>(it, b));
  }
}

}  // namespace cylon
