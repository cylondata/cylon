#include <arrow/compute/api.h>
#include <arrow/api.h>
#include <glog/logging.h>
#include "arrow_utils.hpp"

namespace twisterx {
  namespace util {
	template<typename TYPE>
	arrow::Status do_copy_numeric_array(std::shared_ptr<std::vector<int64_t>> indices,
										std::shared_ptr<arrow::Array> data_array,
										std::shared_ptr<arrow::Array> *copied_array,
										arrow::MemoryPool *memory_pool) {
    arrow::NumericBuilder<TYPE> array_builder(memory_pool);
    /*arrow::Status status = array_builder.Reserve(indices->size());
    if (status != arrow::Status::OK()) {
          LOG(FATAL) << "Failed to reserve memory when re arranging the array based on indices. " << status.ToString();
          return status;
    }*/

    auto casted_array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(data_array);
    for (auto &index : *indices) {
      double x = casted_array->Value(index) + 0;
      if (casted_array->length() <= index) {
        LOG(INFO) << "INVALID INDEX " << index << " LENGTH " << casted_array->length();
      }
      /** arrow::Status status = array_builder.Append(casted_array->Value(index));
           if (status != arrow::Status::OK()) {
             LOG(FATAL) << "Failed to append rearranged data points to the array builder. " << status.ToString();
             return status;
           }*/
    }
    return array_builder.Finish(copied_array);
	}

	arrow::Status copy_array_by_indices(std::shared_ptr<std::vector<int64_t>> indices,
										std::shared_ptr<arrow::Array> data_array,
										std::shared_ptr<arrow::Array> *copied_array,
										arrow::MemoryPool *memory_pool) {
	  switch (data_array->type()->id()) {
		case arrow::Type::NA:break;
		case arrow::Type::BOOL:break;
		case arrow::Type::UINT8:
		  return do_copy_numeric_array<arrow::UInt8Type>(indices,
														 data_array,
														 copied_array,
														 memory_pool);
		case arrow::Type::INT8:
		  return do_copy_numeric_array<arrow::Int8Type>(indices,
														data_array,
														copied_array,
														memory_pool);
		case arrow::Type::UINT16:
		  return do_copy_numeric_array<arrow::Int16Type>(indices,
														 data_array,
														 copied_array,
														 memory_pool);
		case arrow::Type::INT16:
		  return do_copy_numeric_array<arrow::Int16Type>(indices,
														 data_array,
														 copied_array,
														 memory_pool);
		case arrow::Type::UINT32:
		  return do_copy_numeric_array<arrow::UInt32Type>(indices,
														  data_array,
														  copied_array,
														  memory_pool);
		case arrow::Type::INT32:
		  return do_copy_numeric_array<arrow::Int32Type>(indices,
														 data_array,
														 copied_array,
														 memory_pool);
		case arrow::Type::UINT64:
		  return do_copy_numeric_array<arrow::UInt64Type>(indices,
														  data_array,
														  copied_array,
														  memory_pool);
		case arrow::Type::INT64:
		  return do_copy_numeric_array<arrow::Int64Type>(indices,
														 data_array,
														 copied_array,
														 memory_pool);
		case arrow::Type::HALF_FLOAT:
		  return do_copy_numeric_array<arrow::HalfFloatType>(indices,
															 data_array,
															 copied_array,
															 memory_pool);
		case arrow::Type::FLOAT:
		  return do_copy_numeric_array<arrow::FloatType>(indices,
														 data_array,
														 copied_array,
														 memory_pool);
		case arrow::Type::DOUBLE:
		  return do_copy_numeric_array<arrow::DoubleType>(indices,
														  data_array,
														  copied_array,
														  memory_pool);
		case arrow::Type::STRING:break;
		case arrow::Type::BINARY:break;
		case arrow::Type::FIXED_SIZE_BINARY:break;
		case arrow::Type::DATE32:break;
		case arrow::Type::DATE64:break;
		case arrow::Type::TIMESTAMP:break;
		case arrow::Type::TIME32:break;
		case arrow::Type::TIME64:break;
		case arrow::Type::INTERVAL:break;
		case arrow::Type::DECIMAL:break;
		case arrow::Type::LIST:break;
		case arrow::Type::STRUCT:break;
		case arrow::Type::UNION:break;
		case arrow::Type::DICTIONARY:break;
		case arrow::Type::MAP:break;
		case arrow::Type::EXTENSION:break;
		case arrow::Type::FIXED_SIZE_LIST:break;
		case arrow::Type::DURATION:break;
		case arrow::Type::LARGE_STRING:break;
		case arrow::Type::LARGE_BINARY:break;
		case arrow::Type::LARGE_LIST:break;
	  }
	}
  }
}
