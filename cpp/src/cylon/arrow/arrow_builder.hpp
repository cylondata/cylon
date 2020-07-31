#ifndef CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_
#define CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_

#include <string>
#include <vector>
#include "../status.hpp"
namespace cylon {
namespace cyarrow {
cylon::Status BeginTable(const std::string &table_id);
cylon::Status AddColumn(const std::string &table_id, const std::string &col_name, int8_t type,
                        int32_t value_count,
                        int32_t null_count,
                        int64_t validity_address, int64_t validity_size,
                        int64_t data_address, int64_t data_size);
cylon::Status AddColumn(const std::string &table_id, const std::string &col_name, int8_t type,
                        int32_t value_count,
                        int32_t null_count,
                        int64_t validity_address, int64_t validity_size,
                        int64_t data_address, int64_t data_size,
                        int64_t offset_address, int64_t offset_size);
cylon::Status FinishTable(const std::string &table_id);
}
}

#endif //CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_
