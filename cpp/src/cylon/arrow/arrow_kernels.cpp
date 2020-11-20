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
cylon::Status CreateSplitter(const std::shared_ptr<arrow::DataType> &type,
                             arrow::MemoryPool *pool,
                             std::shared_ptr<ArrowArraySplitKernel> *out) {
  switch (type->id()) {
    case arrow::Type::UINT8:*out = std::make_shared<UInt8ArraySplitter>(pool);
      break;
    case arrow::Type::INT8:*out = std::make_shared<Int8ArraySplitter>(pool);
      break;
    case arrow::Type::UINT16:*out = std::make_shared<UInt16ArraySplitter>(pool);
      break;
    case arrow::Type::INT16:*out = std::make_shared<Int16ArraySplitter>(pool);
      break;
    case arrow::Type::UINT32:*out = std::make_shared<UInt32ArraySplitter>(pool);
      break;
    case arrow::Type::INT32:*out = std::make_shared<Int32ArraySplitter>(pool);
      break;
    case arrow::Type::UINT64:*out = std::make_shared<UInt64ArraySplitter>(pool);
      break;
    case arrow::Type::INT64:*out = std::make_shared<Int64ArraySplitter>(pool);
      break;
    case arrow::Type::FLOAT:*out = std::make_shared<FloatArraySplitter>(pool);
      break;
    case arrow::Type::DOUBLE:*out = std::make_shared<DoubleArraySplitter>(pool);
      break;
    case arrow::Type::FIXED_SIZE_BINARY:*out = std::make_shared<FixedBinaryArraySplitKernel>(pool);
      break;
    case arrow::Type::STRING:*out = std::make_shared<BinaryArraySplitKernel<arrow::StringType>>(pool);
      break;
    case arrow::Type::BINARY:*out = std::make_shared<BinaryArraySplitKernel<arrow::BinaryType>>(pool);
      break;
    default:LOG_AND_RETURN_ERROR(Code::NotImplemented, std::string("Un-known type " + type->name()))
  }
  return cylon::Status::OK();
}

int FixedBinaryArraySplitKernel::Split(std::shared_ptr<arrow::Array> &values,
                                       const std::vector<int64_t> &partitions,
                                       const std::vector<int32_t> &targets,
                                       std::unordered_map<int, std::shared_ptr<arrow::Array>> &out,
                                       std::vector<uint32_t> &counts) {
  auto reader = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
  std::unordered_map<int, std::shared_ptr<arrow::FixedSizeBinaryBuilder>> builders;

  for (size_t i = 0; i < targets.size(); i++) {
    int target = targets[i];
    std::shared_ptr<arrow::FixedSizeBinaryBuilder> b =
        std::make_shared<arrow::FixedSizeBinaryBuilder>(values->type(), pool_);
    arrow::Status st = b->Reserve(counts[i]);
    builders.insert(std::pair<int, std::shared_ptr<arrow::FixedSizeBinaryBuilder>>(target, b));
  }

  for (size_t i = 0; i < partitions.size(); i++) {
    std::shared_ptr<arrow::FixedSizeBinaryBuilder> b = builders[partitions.at(i)];
    if (b->Append(reader->Value(i)) != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to merge";
      return -1;
    }
  }

  for (int it : targets) {
    std::shared_ptr<arrow::FixedSizeBinaryBuilder> b = builders[it];
    std::shared_ptr<arrow::Array> array;
    if (b->Finish(&array) != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to merge";
      return -1;
    }
    out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(it, array));
  }
  return 0;
}

Status FixedBinaryArraySplitKernel::Split(const std::shared_ptr<arrow::ChunkedArray> &values,
                                          const std::vector<uint32_t> &target_partitions,
                                          uint32_t num_partitions,
                                          const std::vector<uint32_t> &counts,
                                          std::vector<std::shared_ptr<arrow::Array>> &output) {
  // todo implement this
  return Status();
}

/*int BinaryArraySplitKernel::Split(std::shared_ptr<arrow::Array> &values,
                                  const std::vector<int64_t> &partitions,
                                  const std::vector<int32_t> &targets,
                                  std::unordered_map<int, std::shared_ptr<arrow::Array>> &out,
                                  std::vector<uint32_t> &counts) {
  auto reader =
      std::static_pointer_cast<arrow::BinaryArray>(values);
  std::unordered_map<int, std::shared_ptr<arrow::BinaryBuilder>> builders;

  for (int it : targets) {
    std::shared_ptr<arrow::BinaryBuilder> b = std::make_shared<arrow::BinaryBuilder>(type_, pool_);
    builders.insert(std::pair<int, std::shared_ptr<arrow::BinaryBuilder>>(it, b));
  }

  for (size_t i = 0; i < partitions.size(); i++) {
    std::shared_ptr<arrow::BinaryBuilder> b = builders[partitions.at(i)];
    int length = 0;
    const uint8_t *value = reader->GetValue(i, &length);
    if (b->Append(value, length) != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to merge";
      return -1;
    }
  }

  for (int it : targets) {
    std::shared_ptr<arrow::BinaryBuilder> b = builders[it];
    std::shared_ptr<arrow::Array> array;
    if (b->Finish(&array) != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to merge";
      return -1;
    }
    out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(it, array));
  }
  return 0;
}

Status BinaryArraySplitKernel::Split(const std::shared_ptr<arrow::ChunkedArray> &values,
                                     const std::vector<int32_t> &target_partitions,
                                     int32_t num_partitions,
                                     const std::vector<uint32_t> &counts,
                                     std::vector<std::shared_ptr<arrow::Array>> &output) {
  // todo implement this
  return Status();
}*/

class ArrowStringSortKernel : public ArrowArraySortKernel {
 public:
  explicit ArrowStringSortKernel(arrow::MemoryPool *pool) :
      ArrowArraySortKernel(pool) {}

  int Sort(std::shared_ptr<arrow::Array> values,
           std::shared_ptr<arrow::Array> *offsets) override {
    auto array = std::static_pointer_cast<arrow::StringArray>(values);
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(int64_t);

    arrow::Result<std::unique_ptr<arrow::Buffer>> result = AllocateBuffer(buf_size + 1, pool_);
    const arrow::Status &status = result.status();
    if (!status.ok()) {
      LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return -1;
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
    *offsets = std::make_shared<arrow::UInt64Array>(values->length(), indices_buf);
    return 0;
  }
};

class ArrowFixedSizeBinarySortKernel : public ArrowArraySortKernel {
 public:
  explicit ArrowFixedSizeBinarySortKernel(arrow::MemoryPool *pool) :
      ArrowArraySortKernel(pool) {}

  int Sort(std::shared_ptr<arrow::Array> values,
           std::shared_ptr<arrow::Array> *offsets) override {
    auto array = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(uint64_t);
    arrow::Result<std::unique_ptr<arrow::Buffer>> result = AllocateBuffer(buf_size + 1, pool_);
    const arrow::Status &status = result.status();
    if (!status.ok()) {
      LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return -1;
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
    *offsets = std::make_shared<arrow::UInt64Array>(values->length(), indices_buf);
    return 0;
  }
};

class ArrowBinarySortKernel : public ArrowArraySortKernel {
 public:
  explicit ArrowBinarySortKernel(arrow::MemoryPool *pool) :
      ArrowArraySortKernel(pool) {}

  int Sort(std::shared_ptr<arrow::Array> values,
           std::shared_ptr<arrow::Array> *offsets) override {
    auto array = std::static_pointer_cast<arrow::BinaryArray>(values);
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(uint64_t);
    arrow::Result<std::unique_ptr<arrow::Buffer>> result = AllocateBuffer(buf_size + 1, pool_);
    const arrow::Status &status = result.status();
    if (!status.ok()) {
      LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return -1;
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
    *offsets = std::make_shared<arrow::UInt64Array>(values->length(), indices_buf);
    return 0;
  }
};

int CreateSorter(const std::shared_ptr<arrow::DataType> &type,
                 arrow::MemoryPool *pool,
                 std::shared_ptr<ArrowArraySortKernel> *out) {
  switch (type->id()) {
    case arrow::Type::UINT8:*out = std::make_shared<UInt8ArraySorter>(pool);
      break;
    case arrow::Type::INT8:*out = std::make_shared<Int8ArraySorter>(pool);
      break;
    case arrow::Type::UINT16:*out = std::make_shared<UInt16ArraySorter>(pool);
      break;
    case arrow::Type::INT16:*out = std::make_shared<Int16ArraySorter>(pool);
      break;
    case arrow::Type::UINT32:*out = std::make_shared<UInt32ArraySorter>(pool);
      break;
    case arrow::Type::INT32:*out = std::make_shared<Int32ArraySorter>(pool);
      break;
    case arrow::Type::UINT64:*out = std::make_shared<UInt64ArraySorter>(pool);
      break;
    case arrow::Type::INT64:*out = std::make_shared<Int64ArraySorter>(pool);
      break;
    case arrow::Type::FLOAT:*out = std::make_shared<FloatArraySorter>(pool);
      break;
    case arrow::Type::DOUBLE:*out = std::make_shared<DoubleArraySorter>(pool);
      break;
    case arrow::Type::STRING:*out = std::make_shared<ArrowStringSortKernel>(pool);
      break;
    case arrow::Type::BINARY:*out = std::make_shared<ArrowBinarySortKernel>(pool);
      break;
    case arrow::Type::FIXED_SIZE_BINARY:*out = std::make_shared<ArrowFixedSizeBinarySortKernel>(pool);
      break;
    default:LOG(FATAL) << "Un-known type";
      return -1;
  }
  return 0;
}

arrow::Status SortIndices(arrow::MemoryPool *memory_pool, std::shared_ptr<arrow::Array> &values,
                          std::shared_ptr<arrow::Array> *offsets) {
  std::shared_ptr<ArrowArraySortKernel> out;
  if (CreateSorter(values->type(), memory_pool, &out) != 0) {
    return arrow::Status(arrow::StatusCode::NotImplemented, "unknown type " + values->type()->ToString());
  }
  out->Sort(values, offsets);
  return arrow::Status::OK();
}

int CreateInplaceSorter(const std::shared_ptr<arrow::DataType> &type,
                        arrow::MemoryPool *pool,
                        std::shared_ptr<ArrowArrayInplaceSortKernel> *out) {
  switch (type->id()) {
    case arrow::Type::UINT8:*out = std::make_shared<UInt8ArrayInplaceSorter>(pool);
      break;
    case arrow::Type::INT8:*out = std::make_shared<Int8ArrayInplaceSorter>(pool);
      break;
    case arrow::Type::UINT16:*out = std::make_shared<UInt16ArrayInplaceSorter>(pool);
      break;
    case arrow::Type::INT16:*out = std::make_shared<Int16ArrayInplaceSorter>(pool);
      break;
    case arrow::Type::UINT32:*out = std::make_shared<UInt32ArrayInplaceSorter>(pool);
      break;
    case arrow::Type::INT32:*out = std::make_shared<Int32ArrayInplaceSorter>(pool);
      break;
    case arrow::Type::UINT64:*out = std::make_shared<UInt64ArrayInplaceSorter>(pool);
      break;
    case arrow::Type::INT64:*out = std::make_shared<Int64ArrayInplaceSorter>(pool);
      break;
    case arrow::Type::FLOAT:*out = std::make_shared<FloatArrayInplaceSorter>(pool);
      break;
    case arrow::Type::DOUBLE:*out = std::make_shared<DoubleArrayInplaceSorter>(pool);
      break;
    default:LOG(FATAL) << "Un-known type";
      return -1;
  }
  return 0;
}

arrow::Status SortIndicesInPlace(arrow::MemoryPool *memory_pool,
                                 std::shared_ptr<arrow::Array> &values,
                                 std::shared_ptr<arrow::UInt64Array> *offsets) {
  std::shared_ptr<ArrowArrayInplaceSortKernel> out;
  if (CreateInplaceSorter(values->type(), memory_pool, &out) != 0) {
    return arrow::Status(arrow::StatusCode::NotImplemented, "unknown type " + values->type()->ToString());
  }
  out->Sort(values, offsets);
  return arrow::Status::OK();
}

}  // namespace cylon
