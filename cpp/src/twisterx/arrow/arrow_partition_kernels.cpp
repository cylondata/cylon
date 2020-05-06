#include "arrow_partition_kernels.hpp"

namespace twisterx {

ArrowPartitionKernel *GetPartitionKernel(arrow::MemoryPool *pool,
										 std::shared_ptr<arrow::Array> values) {
  ArrowPartitionKernel *kernel;
  switch (values->type()->id()) {
	case arrow::Type::UINT8:kernel = new UInt8ArrayHashPartitioner(pool);
	  break;
	case arrow::Type::INT8:kernel = new Int8ArrayHashPartitioner(pool);
	  break;
	case arrow::Type::UINT16:kernel = new UInt16ArrayHashPartitioner(pool);
	  break;
	case arrow::Type::INT16:kernel = new Int16ArrayHashPartitioner(pool);
	  break;
	case arrow::Type::UINT32:kernel = new UInt32ArrayHashPartitioner(pool);
	  break;
	case arrow::Type::INT32:kernel = new Int32ArrayHashPartitioner(pool);
	  break;
	case arrow::Type::UINT64:kernel = new UInt64ArrayHashPartitioner(pool);
	  break;
	case arrow::Type::INT64:kernel = new Int64ArrayHashPartitioner(pool);
	  break;
	case arrow::Type::FLOAT:kernel = new FloatArrayHashPartitioner(pool);
	  break;
	case arrow::Type::DOUBLE:kernel = new DoubleArrayHashPartitioner(pool);
	  break;
	default:LOG(FATAL) << "Un-known type";
	  return NULLPTR;
  }
  return kernel;
}

twisterx::Status HashPartitionArray(arrow::MemoryPool *pool,
									std::shared_ptr<arrow::Array> values,
									const std::vector<int> &targets,
									std::vector<int64_t> *outPartitions) {
  ArrowPartitionKernel *kernel = GetPartitionKernel(pool, values);
  kernel->Partition(values, targets, outPartitions);
  return twisterx::Status::OK();
}

twisterx::Status HashPartitionArrays(arrow::MemoryPool *pool,
									 std::vector<std::shared_ptr<arrow::Array>> values,
									 int64_t length,
									 const std::vector<int> &targets,
									 std::vector<int64_t> *outPartitions) {
  std::vector<ArrowPartitionKernel *> hash_kernels;
  for (const auto &array: values) {
	auto hash_kernel = GetPartitionKernel(pool, array);
	if (hash_kernel == NULLPTR) {
	  LOG(FATAL) << "Un-known type";
	  return twisterx::Status(twisterx::NotImplemented, "Not implemented or unsupported data type.");
	}
	hash_kernels.push_back(hash_kernel);
  }

  for (int64_t index = 0; index < length; index++) {
	int64_t hash_code = 1;
	int64_t array_index = 0;
	for (const auto &array: values) {
	  hash_code = 31 * hash_code + hash_kernels[array_index++]->ToHash(array, index);
	}
	outPartitions->push_back(targets[hash_code % targets.size()]);
  }
  return twisterx::Status::OK();
}

}
