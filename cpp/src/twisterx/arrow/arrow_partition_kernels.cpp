#include "arrow_partition_kernels.hpp"

namespace twisterx {

arrow::Status HashPartitionArray(std::shared_ptr<arrow::DataType>& type, arrow::MemoryPool *pool,
                                 std::shared_ptr<arrow::Array> values,
                                 std::shared_ptr<std::vector<int>> targets,
                                 std::shared_ptr<arrow::Array>* outPartitions) {
  ArrowPartitionKernel* kernel;
  switch (type->id()) {
    case arrow::Type::UINT8:
      kernel = new UInt8ArrayHashPartitioner(type, pool, targets);
      break;
    case arrow::Type::INT8:
      kernel = new Int8ArrayHashPartitioner(type, pool, targets);
      break;
    case arrow::Type::UINT16:
      kernel = new UInt16ArrayHashPartitioner(type, pool, targets);
      break;
    case arrow::Type::INT16:
      kernel = new Int16ArrayHashPartitioner(type, pool, targets);
      break;
    case arrow::Type::UINT32:
      kernel = new UInt32ArrayHashPartitioner(type, pool, targets);
      break;
    case arrow::Type::INT32:
      kernel = new Int32ArrayHashPartitioner(type, pool, targets);
      break;
    case arrow::Type::UINT64:
      kernel = new UInt64ArrayHashPartitioner(type, pool, targets);
      break;
    case arrow::Type::INT64:
      kernel = new Int64ArrayHashPartitioner(type, pool, targets);
      break;
    case arrow::Type::FLOAT:
      kernel = new FloatArrayHashPartitioner(type, pool, targets);
      break;
    case arrow::Type::DOUBLE:
      kernel = new DoubleArrayHashPartitioner(type, pool, targets);
      break;
    default:
      LOG(FATAL) << "Un-known type";
      return arrow::Status::NotImplemented("Not implemented");
  }
  kernel->Partition(values, outPartitions);
  return arrow::Status::OK();
}

}
