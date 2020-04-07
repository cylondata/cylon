#include "arrow_types.hpp"
#include "../data_types.hpp"

namespace twisterx {
namespace tarrow {

std::shared_ptr<arrow::DataType> convertToArrowType(std::shared_ptr<DataType> tType) {
  if (tType->getType() == Type::INT8) {
  }
  return nullptr;
}

}
}