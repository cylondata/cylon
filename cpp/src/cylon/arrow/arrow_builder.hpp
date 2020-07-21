#ifndef CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_
#define CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_

#include <string>
#include <vector>
namespace cylon {
namespace cyarrow {
void BeginTable(const std::string& table_id);
void AddColumn(const std::string& table_id, int32_t col_index, int32_t type, int64_t address, int64_t size);
void EndTable(std::string table_id);
}
}

#endif //CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_
