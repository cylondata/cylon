#include "column.hpp"

namespace twisterx {

std::shared_ptr<Column> twisterx::Column::FromArrow(std::shared_ptr<arrow::Array> array) {
  return std::shared_ptr<Column>();
}

}
