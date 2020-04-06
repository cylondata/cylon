#include "arrow_kernels.hpp"

namespace twisterx {
int CreateNumericMerge(std::shared_ptr<arrow::DataType>& type,
                       arrow::MemoryPool* pool, std::shared_ptr<std::vector<int>> targets,
                       std::unique_ptr<ArrowArrayMergeKernel>* out) {
  ArrowArrayMergeKernel* kernel;
  switch (type->id()) {
    case arrow::Type::UINT8:
      kernel = new UInt8ArrayMerger(type, pool, targets);
      break;
    case arrow::Type::INT8:
      kernel = new Int8ArrayMerger(type, pool, targets);
      break;
    case arrow::Type::UINT16:
      kernel = new UInt16ArrayMerger(type, pool, targets);
      break;
    case arrow::Type::INT16:
      kernel = new Int16ArrayMerger(type, pool, targets);
      break;
    case arrow::Type::UINT32:
      kernel = new UInt32ArrayMerger(type, pool, targets);
      break;
    case arrow::Type::INT32:
      kernel = new Int32ArrayMerger(type, pool, targets);
      break;
    case arrow::Type::UINT64:
      kernel = new UInt64ArrayMerger(type, pool, targets);
      break;
    case arrow::Type::INT64:
      kernel = new Int64ArrayMerger(type, pool, targets);
      break;
    case arrow::Type::FLOAT:
      kernel = new FloatArrayMerger(type, pool, targets);
      break;
    case arrow::Type::DOUBLE:
      kernel = new DoubleArrayMerger(type, pool, targets);
      break;
    case arrow::Type::FIXED_SIZE_BINARY:
      kernel = new FixedBinaryArrayMerger(type, pool, targets);
      break;
    case arrow::Type::BINARY:
      kernel = new BinaryArrayMerger(type, pool, targets);
      break;
    default:
      //LOG(FATAL) << "Un-known type";
      return -1;
  }
  out->reset(kernel);
  return 0;
}

int FixedBinaryArrayMerger::Merge(std::shared_ptr<arrow::Array> &values,
    std::shared_ptr<arrow::Int32Array> &partitions,
    std::unordered_map<int, std::shared_ptr<arrow::Array> > &out) {
  auto reader =
      std::static_pointer_cast<arrow::FixedSizeBinaryArray>(values);
  std::unordered_map<int, std::shared_ptr<arrow::FixedSizeBinaryBuilder>> builders;

  for (int & it : *targets_) {
    std::shared_ptr<arrow::FixedSizeBinaryBuilder> b = std::make_shared<arrow::FixedSizeBinaryBuilder>(type_, pool_);
    builders.insert(std::pair<int, std::shared_ptr<arrow::FixedSizeBinaryBuilder>>(it, b));
  }

  for (int64_t i = 0; i < partitions->length(); i++) {
    std::shared_ptr<arrow::FixedSizeBinaryBuilder> b = builders[partitions->Value(i)];
    if (b->Append(reader->Value(i)) != arrow::Status::OK()) {
      //LOG(FATAL) << "Failed to merge";
      return -1;
    }
  }

  for (int & it : *targets_) {
    std::shared_ptr<arrow::FixedSizeBinaryBuilder> b = builders[it];
    std::shared_ptr<arrow::Array> array;
    if (b->Finish(&array) != arrow::Status::OK()) {
      //LOG(FATAL) << "Failed to merge";
      return -1;
    }
    out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(it, array));
  }
  return 0;
}

int BinaryArrayMerger::Merge(std::shared_ptr<arrow::Array> &values,
    std::shared_ptr<arrow::Int32Array> &partitions,
    std::unordered_map<int, std::shared_ptr<arrow::Array> > &out) {
  auto reader =
      std::static_pointer_cast<arrow::BinaryArray>(values);
  std::unordered_map<int, std::shared_ptr<arrow::BinaryBuilder>> builders;

  for (int & it : *targets_) {
    std::shared_ptr<arrow::BinaryBuilder> b = std::make_shared<arrow::BinaryBuilder>(type_, pool_);
    builders.insert(std::pair<int, std::shared_ptr<arrow::BinaryBuilder>>(it, b));
  }

  for (int64_t i = 0; i < partitions->length(); i++) {
    std::shared_ptr<arrow::BinaryBuilder> b = builders[partitions->Value(i)];
    int length = 0;
    const uint8_t *value = reader->GetValue(i, &length);
    if (b->Append(value, length) != arrow::Status::OK()) {
      //LOG(FATAL) << "Failed to merge";
      return -1;
    }
  }

  for (int & it : *targets_) {
    std::shared_ptr<arrow::BinaryBuilder> b = builders[it];
    std::shared_ptr<arrow::Array> array;
    if (b->Finish(&array) != arrow::Status::OK()) {
      //LOG(FATAL) << "Failed to merge";
      return -1;
    }
    out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(it, array));
  }
  return 0;
}

class ArrowStringSortKernel : public ArrowArraySortKernel {
public:
  explicit ArrowStringSortKernel(std::shared_ptr<arrow::DataType> type,
                                 arrow::MemoryPool* pool) :
      ArrowArraySortKernel(type, pool) {}

  int Sort(std::shared_ptr<arrow::Array> values,
           std::shared_ptr<arrow::Array>* offsets) override {
    auto array = std::static_pointer_cast<arrow::StringArray>(values);
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(uint64_t);
    arrow::Status status = AllocateBuffer(arrow::default_memory_pool(), buf_size + 1, &indices_buf);
    if (status != arrow::Status::OK()) {
      //LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return -1;
    }
    auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
    for (int64_t i = 0; i < values->length(); i++) {
      indices_begin[i] = i;
    }
    int64_t *indices_end = indices_begin + values->length();
    std::sort(indices_begin, indices_end, [array](uint64_t left, uint64_t right) {
      return array->GetView(left).compare(array->GetView(right));
    });
    *offsets = std::make_shared<arrow::UInt64Array>(values->length(), indices_buf);
    return 0;
  }
};

class ArrowBinarySortKernel : public ArrowArraySortKernel {
public:
  explicit ArrowBinarySortKernel(std::shared_ptr<arrow::DataType> type,
                                 arrow::MemoryPool* pool) :
      ArrowArraySortKernel(type, pool) {}

  int Sort(std::shared_ptr<arrow::Array> values,
           std::shared_ptr<arrow::Array>* offsets) override {
    auto array = std::static_pointer_cast<arrow::BinaryArray>(values);
    std::shared_ptr<arrow::Buffer> indices_buf;
    int64_t buf_size = values->length() * sizeof(uint64_t);
    arrow::Status status = AllocateBuffer(arrow::default_memory_pool(), buf_size + 1, &indices_buf);
    if (status != arrow::Status::OK()) {
      //LOG(FATAL) << "Failed to allocate sort indices - " << status.message();
      return -1;
    }
    auto *indices_begin = reinterpret_cast<int64_t *>(indices_buf->mutable_data());
    for (int64_t i = 0; i < values->length(); i++) {
      indices_begin[i] = i;
    }
    int64_t *indices_end = indices_begin + values->length();
    std::sort(indices_begin, indices_end, [array](uint64_t left, uint64_t right) {
      return array->GetView(left).compare(array->GetView(right));
    });
    *offsets = std::make_shared<arrow::UInt64Array>(values->length(), indices_buf);
    return 0;
  }
};

int CreateSorter(std::shared_ptr<arrow::DataType> type,
                       arrow::MemoryPool* pool,
                       std::unique_ptr<ArrowArraySortKernel>* out) {
  ArrowArraySortKernel* kernel;
  switch (type->id()) {
    case arrow::Type::UINT8:
      kernel = new UInt8ArraySorter(type, pool);
      break;
    case arrow::Type::INT8:
      kernel = new Int8ArraySorter(type, pool);
      break;
    case arrow::Type::UINT16:
      kernel = new UInt16ArraySorter(type, pool);
      break;
    case arrow::Type::INT16:
      kernel = new Int16ArraySorter(type, pool);
      break;
    case arrow::Type::UINT32:
      kernel = new UInt32ArraySorter(type, pool);
      break;
    case arrow::Type::INT32:
      kernel = new Int32ArraySorter(type, pool);
      break;
    case arrow::Type::UINT64:
      kernel = new UInt64ArraySorter(type, pool);
      break;
    case arrow::Type::INT64:
      kernel = new Int64ArraySorter(type, pool);
      break;
    case arrow::Type::FLOAT:
      kernel = new FloatArraySorter(type, pool);
      break;
    case arrow::Type::DOUBLE:
      kernel = new DoubleArraySorter(type, pool);
      break;
    case arrow::Type::STRING:
      kernel = new ArrowStringSortKernel(type, pool);
      break;
    case arrow::Type::BINARY:
      kernel = new ArrowBinarySortKernel(type, pool);
      break;
    case arrow::Type::FIXED_SIZE_BINARY:
      kernel = new ArrowBinarySortKernel(type, pool);
      break;
    default:
      //LOG(FATAL) << "Un-known type";
      return -1;
  }
  out->reset(kernel);
  return 0;
}

arrow::Status SortIndices(arrow::MemoryPool *memory_pool, std::shared_ptr<arrow::Array> values,
                std::shared_ptr<arrow::Array>* offsets) {
  std::unique_ptr<ArrowArraySortKernel> out;
  CreateSorter(values->type(), memory_pool, &out);
  out->Sort(values, offsets);
  return arrow::Status::OK();
}

}