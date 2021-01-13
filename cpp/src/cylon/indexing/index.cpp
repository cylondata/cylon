#include "index.hpp"
#include "table.hpp"

namespace cylon {

std::unique_ptr<IndexKernel> CreateHashIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column) {
  switch (input_table->column(index_column)->chunk(0)->type()->id()) {

    case arrow::Type::NA:return nullptr;
    case arrow::Type::BOOL:return std::make_unique<BoolHashIndexKernel>();
    case arrow::Type::UINT8:return std::make_unique<UInt8HashIndexKernel>();
    case arrow::Type::INT8:return std::make_unique<Int8HashIndexKernel>();
    case arrow::Type::UINT16:return std::make_unique<UInt16HashIndexKernel>();
    case arrow::Type::INT16:return std::make_unique<Int16HashIndexKernel>();
    case arrow::Type::UINT32:return std::make_unique<UInt32HashIndexKernel>();
    case arrow::Type::INT32:return std::make_unique<Int32HashIndexKernel>();
    case arrow::Type::UINT64:return std::make_unique<UInt64HashIndexKernel>();
    case arrow::Type::INT64: return std::make_unique<Int64HashIndexKernel>();
    case arrow::Type::HALF_FLOAT:return std::make_unique<HalfFloatHashIndexKernel>();
    case arrow::Type::FLOAT:return std::make_unique<FloatHashIndexKernel>();
    case arrow::Type::DOUBLE:return std::make_unique<DoubleHashIndexKernel>();
    case arrow::Type::STRING:return nullptr;
    case arrow::Type::BINARY:return nullptr;
    default: return nullptr;
  }

  //TODO : returning nullptr issue
  return std::unique_ptr<IndexKernel>();


}

}



