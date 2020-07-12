#ifndef CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_
#define CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_

#include <string>
#include <vector>
namespace cylon {
namespace carrow {
void Build(std::string table_id,
           uint8_t *schema,
           int64_t schema_length,
           std::vector<int8_t *> buffers,
           std::vector<int64_t> lengths);
}
}

#endif //CYLON_SRC_CYLON_ARROW_ARROW_BUILDER_HPP_
