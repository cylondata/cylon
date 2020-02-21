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
        LOG(FATAL) << "Un-known type";
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
      b->Append(reader->Value(i));
    }

    for (int & it : *targets_) {
      std::shared_ptr<arrow::FixedSizeBinaryBuilder> b = builders[it];
      std::shared_ptr<arrow::Array> array;
      b->Finish(&array);
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
      b->Append(value, length);
    }

    for (int & it : *targets_) {
      std::shared_ptr<arrow::BinaryBuilder> b = builders[it];
      std::shared_ptr<arrow::Array> array;
      b->Finish(&array);
      out.insert(std::pair<int, std::shared_ptr<arrow::Array>>(it, array));
    }
    return 0;
  }
}