#include "arrow_types.hpp"
#include "../data_types.hpp"

namespace twisterx {
namespace tarrow {

std::shared_ptr<arrow::DataType> convertToArrowType(std::shared_ptr<DataType> tType) {
  if (tType->getType() == Type::INT8) {
  }
  return nullptr;
}

bool validateArrowTableTypes(const std::shared_ptr<arrow::Table>& table) {
  std::shared_ptr<arrow::Schema> schema = table->schema();
  for (const auto &t : schema->fields()) {
    switch (t->type()->id()) {
      case arrow::Type::NA:
        break;
      case arrow::Type::BOOL:
        break;
      case arrow::Type::UINT8:
      case arrow::Type::INT8:
      case arrow::Type::UINT16:
      case arrow::Type::INT16:
      case arrow::Type::UINT32:
      case arrow::Type::INT32:
      case arrow::Type::UINT64:
      case arrow::Type::INT64:
      case arrow::Type::HALF_FLOAT:
      case arrow::Type::FLOAT:
      case arrow::Type::DOUBLE:
      case arrow::Type::BINARY:
      case arrow::Type::FIXED_SIZE_BINARY:
        return true;
      case arrow::Type::STRING:
        break;
      case arrow::Type::DATE32:
        break;
      case arrow::Type::DATE64:
        break;
      case arrow::Type::TIMESTAMP:
        break;
      case arrow::Type::TIME32:
        break;
      case arrow::Type::TIME64:
        break;
      case arrow::Type::INTERVAL:
        break;
      case arrow::Type::DECIMAL:
        break;
      case arrow::Type::LIST: {
        auto t_value = std::static_pointer_cast<arrow::ListType>(t->type());
        switch (t_value->value_type()->id()) {
          case arrow::Type::UINT8:
          case arrow::Type::INT8:
          case arrow::Type::UINT16:
          case arrow::Type::INT16:
          case arrow::Type::UINT32:
          case arrow::Type::INT32:
          case arrow::Type::UINT64:
          case arrow::Type::INT64:
          case arrow::Type::HALF_FLOAT:
          case arrow::Type::FLOAT:
          case arrow::Type::DOUBLE:
            return true;
          default:
            return false;
        }
      }
      case arrow::Type::STRUCT:
      case arrow::Type::UNION:
      case arrow::Type::DICTIONARY:
      case arrow::Type::MAP:
      case arrow::Type::EXTENSION:
      case arrow::Type::FIXED_SIZE_LIST:
      case arrow::Type::DURATION:
      case arrow::Type::LARGE_STRING:
      case arrow::Type::LARGE_BINARY:
      case arrow::Type::LARGE_LIST:
        return false;
    }
  }
}

}
}