#ifndef TWISTERX_SRC_TWISTERX_TABLE_API_EXTENDED_HPP_
#define TWISTERX_SRC_TWISTERX_TABLE_API_EXTENDED_HPP_

#include <arrow/api.h>

namespace twisterx {
std::shared_ptr<arrow::Table> GetTable(const std::string &id);
void PutTable(const std::string &id, const std::shared_ptr<arrow::Table> &table);
}
#endif //TWISTERX_SRC_TWISTERX_TABLE_API_EXTENDED_HPP_
