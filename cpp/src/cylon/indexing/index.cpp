
#include "index.hpp"

namespace cylon {

std::unique_ptr<IndexKernel> CreateHashIndexKernel(std::shared_ptr<arrow::Table> input_table, int index_column) {
  switch (input_table->column(index_column)->chunk(0)->type()->id()) {

    case arrow::Type::NA:return nullptr;
    case arrow::Type::BOOL:return nullptr;
    case arrow::Type::UINT8:return nullptr;
    case arrow::Type::INT8:return nullptr;
    case arrow::Type::UINT16:return nullptr;
    case arrow::Type::INT16:return nullptr;
    case arrow::Type::UINT32:return nullptr;
    case arrow::Type::INT32:return nullptr;
    case arrow::Type::UINT64:return nullptr;
    case arrow::Type::INT64: return std::make_unique<Int64HashIndexKernel>();
    case arrow::Type::HALF_FLOAT:return nullptr;
    case arrow::Type::FLOAT:return nullptr;
    case arrow::Type::DOUBLE:return nullptr;
    case arrow::Type::STRING:return nullptr;
    case arrow::Type::BINARY:return nullptr;
    default: return nullptr;
  }
  //TODO : returning nullptr issue
  return std::unique_ptr<IndexKernel>();
}
}