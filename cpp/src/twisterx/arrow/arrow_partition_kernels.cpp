#include "arrow_partition_kernels.hpp"

namespace twisterx {

arrow::Status HashPartitionArray(arrow::MemoryPool *pool,
                                 std::shared_ptr<arrow::Array> values,
                                 const std::vector<int> &targets,
                                 std::vector<int64_t> *outPartitions) {
  ArrowPartitionKernel* kernel;
  switch (values->type()->id()) {
    case arrow::Type::UINT8:
      kernel = new UInt8ArrayHashPartitioner(pool);
      break;
    case arrow::Type::INT8:
      kernel = new Int8ArrayHashPartitioner(pool);
      break;
    case arrow::Type::UINT16:
      kernel = new UInt16ArrayHashPartitioner(pool);
      break;
    case arrow::Type::INT16:
      kernel = new Int16ArrayHashPartitioner(pool);
      break;
    case arrow::Type::UINT32:
      kernel = new UInt32ArrayHashPartitioner(pool);
      break;
    case arrow::Type::INT32:
      kernel = new Int32ArrayHashPartitioner(pool);
      break;
    case arrow::Type::UINT64:
      kernel = new UInt64ArrayHashPartitioner(pool);
      break;
    case arrow::Type::INT64:
      kernel = new Int64ArrayHashPartitioner(pool);
      break;
    case arrow::Type::FLOAT:
      kernel = new FloatArrayHashPartitioner(pool);
      break;
    case arrow::Type::DOUBLE:
      kernel = new DoubleArrayHashPartitioner(pool);
      break;
    default:
      LOG(FATAL) << "Un-known type";
      return arrow::Status::NotImplemented("Not implemented");
  }
  kernel->Partition(values, targets, outPartitions);
  return arrow::Status::OK();
}

}
