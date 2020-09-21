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
  ArrowArraySplitKernel *kernel;
  switch (type->id()) {
    case arrow::Type::UINT8:kernel = new UInt8ArraySplitter(type, pool);
      break;
    case arrow::Type::INT8:kernel = new Int8ArraySplitter(type, pool);
      break;
    case arrow::Type::UINT16:kernel = new UInt16ArraySplitter(type, pool);
      break;
    case arrow::Type::INT16:kernel = new Int16ArraySplitter(type, pool);
      break;
    case arrow::Type::UINT32:kernel = new UInt32ArraySplitter(type, pool);
      break;
    case arrow::Type::INT32:kernel = new Int32ArraySplitter(type, pool);
      break;
    case arrow::Type::UINT64:kernel = new UInt64ArraySplitter(type, pool);
      break;
    case arrow::Type::INT64:kernel = new Int64ArraySplitter(type, pool);
      break;
    case arrow::Type::FLOAT:kernel = new FloatArraySplitter(type, pool);
      break;
    case arrow::Type::DOUBLE:kernel = new DoubleArraySplitter(type, pool);
      break;
    case arrow::Type::FIXED_SIZE_BINARY:kernel = new FixedBinaryArraySplitKernel(type, pool);
      break;
    case arrow::Type::STRING:kernel = new BinaryArraySplitKernel(type, pool);
      break;
    case arrow::Type::BINARY:kernel = new BinaryArraySplitKernel(type, pool);
      break;
    default:
      LOG(FATAL) << "Un-known type " << type->name();
      return cylon::Status(cylon::NotImplemented, "This type not implemented");
  }
  out->reset(kernel);
  return cylon::Status::OK();
}

int FixedBinaryArraySplitKernel::Split(std::shared_ptr<arrow::Array> &values,
                                       const std::vector<int64_t> &partitions,
                                       const std::vector<int32_t> &targets,
                                       std::unordered_map<int, std::shared_ptr<arrow::Array>> &out,
                                       std::vector<uint32_t> &counts) {
  auto reader =
      std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
  std::unordered_map<int, std::shared_ptr<arrow::FixedSizeBinaryBuilder>> builders;

  for (size_t i = 0; i < targets.size(); i++) {
    int target = targets[i];
    std::shared_ptr<arrow::FixedSizeBinaryBuilder> b =
        std::make_shared<arrow::FixedSizeBinaryBuilder>(type_, pool_);
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

int BinaryArraySplitKernel::Split(std::shared_ptr<arrow::Array> &values,
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

class ArrowStringSortKernel : public ArrowArraySortKernel {
 public:
  explicit ArrowStringSortKernel(std::shared_ptr<arrow::DataType> type,
                                 arrow::MemoryPool *pool) :
      ArrowArraySortKernel(type, pool) {}

  int Sort(std::shared_ptr<arrow::Array> values,
           std::shared_ptr<arrow::Array> *offsets) override {
    auto array = std::static_pointer_cast<arrow::StringArray>(values);
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(int64_t);
    arrow::Status status = AllocateBuffer(pool_, buf_size, &indices_buf);
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return -1;
    }
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
  explicit ArrowFixedSizeBinarySortKernel(std::shared_ptr<arrow::DataType> type,
                                 arrow::MemoryPool *pool) :
      ArrowArraySortKernel(std::move(type), pool) {}

  int Sort(std::shared_ptr<arrow::Array> values,
           std::shared_ptr<arrow::Array> *offsets) override {
    auto array = std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(uint64_t);
    arrow::Status status = AllocateBuffer(pool_,
                                          buf_size + 1, &indices_buf);
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return -1;
    }
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
  explicit ArrowBinarySortKernel(std::shared_ptr<arrow::DataType> type,
                                 arrow::MemoryPool *pool) :
      ArrowArraySortKernel(std::move(type), pool) {}

  int Sort(std::shared_ptr<arrow::Array> values,
           std::shared_ptr<arrow::Array> *offsets) override {
    auto array = std::static_pointer_cast<arrow::BinaryArray>(values);
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(uint64_t);
    arrow::Status status = AllocateBuffer(pool_,
        buf_size + 1, &indices_buf);
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return -1;
    }
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

int CreateSorter(std::shared_ptr<arrow::DataType> type,
                 arrow::MemoryPool *pool,
                 std::shared_ptr<ArrowArraySortKernel> *out) {
  ArrowArraySortKernel *kernel;
  switch (type->id()) {
    case arrow::Type::UINT8:kernel = new UInt8ArraySorter(type, pool);
      break;
    case arrow::Type::INT8:kernel = new Int8ArraySorter(type, pool);
      break;
    case arrow::Type::UINT16:kernel = new UInt16ArraySorter(type, pool);
      break;
    case arrow::Type::INT16:kernel = new Int16ArraySorter(type, pool);
      break;
    case arrow::Type::UINT32:kernel = new UInt32ArraySorter(type, pool);
      break;
    case arrow::Type::INT32:kernel = new Int32ArraySorter(type, pool);
      break;
    case arrow::Type::UINT64:kernel = new UInt64ArraySorter(type, pool);
      break;
    case arrow::Type::INT64:kernel = new Int64ArraySorter(type, pool);
      break;
    case arrow::Type::FLOAT:kernel = new FloatArraySorter(type, pool);
      break;
    case arrow::Type::DOUBLE:kernel = new DoubleArraySorter(type, pool);
      break;
    case arrow::Type::STRING:kernel = new ArrowStringSortKernel(type, pool);
      break;
    case arrow::Type::BINARY:kernel = new ArrowBinarySortKernel(type, pool);
      break;
    case arrow::Type::FIXED_SIZE_BINARY:kernel = new ArrowFixedSizeBinarySortKernel(type, pool);
      break;
    default:LOG(FATAL) << "Un-known type";
      return -1;
  }
  out->reset(kernel);
  return 0;
}

arrow::Status SortIndices(arrow::MemoryPool *memory_pool, std::shared_ptr<arrow::Array> values,
                          std::shared_ptr<arrow::Array> *offsets) {
  std::shared_ptr<ArrowArraySortKernel> out;
  CreateSorter(values->type(), memory_pool, &out);
  out->Sort(values, offsets);
  return arrow::Status::OK();
}

int CreateInplaceSorter(std::shared_ptr<arrow::DataType> type,
                        arrow::MemoryPool *pool,
                        std::shared_ptr<ArrowArrayInplaceSortKernel> *out) {
  ArrowArrayInplaceSortKernel *kernel;
  switch (type->id()) {
    case arrow::Type::UINT8:kernel = new UInt8ArrayInplaceSorter(type, pool);
      break;
    case arrow::Type::INT8:kernel = new Int8ArrayInplaceSorter(type, pool);
      break;
    case arrow::Type::UINT16:kernel = new UInt16ArrayInplaceSorter(type, pool);
      break;
    case arrow::Type::INT16:kernel = new Int16ArrayInplaceSorter(type, pool);
      break;
    case arrow::Type::UINT32:kernel = new UInt32ArrayInplaceSorter(type, pool);
      break;
    case arrow::Type::INT32:kernel = new Int32ArrayInplaceSorter(type, pool);
      break;
    case arrow::Type::UINT64:kernel = new UInt64ArrayInplaceSorter(type, pool);
      break;
    case arrow::Type::INT64:kernel = new Int64ArrayInplaceSorter(type, pool);
      break;
    case arrow::Type::FLOAT:kernel = new FloatArrayInplaceSorter(type, pool);
      break;
    case arrow::Type::DOUBLE:kernel = new DoubleArrayInplaceSorter(type, pool);
      break;
    default:LOG(FATAL) << "Un-known type";
      return -1;
  }
  out->reset(kernel);
  return 0;
}

arrow::Status SortIndicesInPlace(arrow::MemoryPool *memory_pool,
                                 std::shared_ptr<arrow::Array> values,
                                 std::shared_ptr<arrow::UInt64Array> *offsets) {
  std::shared_ptr<ArrowArrayInplaceSortKernel> out;
  CreateInplaceSorter(values->type(), memory_pool, &out);
  out->Sort(values, offsets);
  return arrow::Status::OK();
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
