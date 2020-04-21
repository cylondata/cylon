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
std::shared_ptr<arrow::DataType> convertToArrowType(std::shared_ptr<DataType> tType);

/**
 * Validate the types of an arrow table
 * @param table true if we support the types
 * @return false if we don't support the types
 */
bool validateArrowTableTypes(const std::shared_ptr<arrow::Table> &table);

}
}

#endif //TWISTERX_ARROW_TYPES_H
