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
      default:
        LOG(FATAL) << "Un-known type";
        return -1;
    }
    out->reset(kernel);
    return 0;
  }
}