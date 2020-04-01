#ifndef TWISTERX_SRC_IO_TABLE_API_H_
#define TWISTERX_SRC_IO_TABLE_API_H_
#include <string>
#include "Status.h"

namespace twisterx {
namespace io {
twisterx::io::Status read_csv(const std::string &path, const std::string &id);
twisterx::io::Status join(const std::string &table_left,
                          const std::string &table_right,
                          int left_col_idx,
                          int right_col_idx,
                          const std::string &dest_id);
int column_count(const std::string &id);
int row_count(const std::string &id);
twisterx::io::Status print(const std::string &table_id, int col1, int col2, int row1, int row2);
}
}
#endif //TWISTERX_SRC_IO_TABLE_API_H_
