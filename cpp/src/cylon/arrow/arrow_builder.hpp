#ifndef CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_
#define CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_

#include <string>
#include <vector>
#include "../status.hpp"
namespace cylon {
namespace cyarrow {
cylon::Status BeginTable(const std::string &table_id);
cylon::Status AddColumn(const std::string &table_id, const std::string &col_name, int32_t type, int64_t address, int64_t size);
cylon::Status EndTable(const std::string &table_id);
}
}

#endif //CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_
