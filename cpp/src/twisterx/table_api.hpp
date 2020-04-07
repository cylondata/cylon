#ifndef TWISTERX_SRC_IO_TABLE_API_H_
#define TWISTERX_SRC_IO_TABLE_API_H_
#include <string>
#include <vector>
#include "status.hpp"

namespace twisterx {

twisterx::Status read_csv(const std::string &path, const std::string &id);
twisterx::Status joinTables(const std::string &table_left,
                            const std::string &table_right,
                            int left_col_idx,
                            int right_col_idx,
                            const std::string &dest_id);
int column_count(const std::string &id);
int row_count(const std::string &id);
twisterx::Status print(const std::string &table_id, int col1, int col2, int row1, int row2);
twisterx::Status merge(std::vector<std::string> table_ids, const std::string& merged_tab);

}
#endif //TWISTERX_SRC_IO_TABLE_API_H_
