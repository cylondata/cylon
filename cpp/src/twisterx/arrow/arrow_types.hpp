#ifndef TWISTERX_ARROW_TYPES_H
#define TWISTERX_ARROW_TYPES_H

#include <memory>
#include <arrow/api.h>

#include "../data_types.hpp"

namespace twisterx {
namespace tarrow {

/**
 * Convert a twisterx type to an arrow type
 * @param tType the twisterx type
 * @return corresponding arrow type
 */
std::shared_ptr<arrow::DataType> convertToArrowType(DataType tType);

}
}

#endif //TWISTERX_ARROW_TYPES_H
