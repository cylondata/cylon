#include "table_api.h"
#include <memory>
#include <arrow/api.h>
#include <map>
#include "arrow_io.h"
#include "../join/join.hpp"
#include  "../util/to_string.hpp"
#include "iostream"

namespace twisterx {
namespace io {

std::map<std::string, std::shared_ptr<arrow::Table>> table_map{}; //todo make this un ordered

std::shared_ptr<arrow::Table> get_table(const std::string &id) {
  auto itr = table_map.find(id);
  if (itr != table_map.end()) {
    return itr->second;
  }
  return NULLPTR;
}

void put_table(const std::string &id, const std::shared_ptr<arrow::Table> &table) {
  std::pair<std::string, std::shared_ptr<arrow::Table>> pair(id, table);
  table_map.insert(pair);
}

twisterx::io::Status read_csv(const std::string &path, const std::string &id) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = twisterx::io::read_csv(path);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> table = *result;
    put_table(id, table);
    return twisterx::io::Status(Code::OK, result.status().message());
  }
  return twisterx::io::Status(Code::IOError, result.status().message());;
}

twisterx::io::Status print(const std::string &table_id, int col1, int col2, int row1, int row2) {
  auto table = get_table(table_id);
  if (table != NULLPTR) {
    for (int row = row1; row < row2; row++) {
      std::cout << "[";
      for (int col = col1; col < col2; col++) {
        auto column = table->column(col);
        int rowCount = 0;
        for (int chunk = 0; chunk < column->num_chunks(); chunk++) {
          auto array = column->chunk(chunk);
          if (rowCount <= row && rowCount + array->length() > row) {
            // print this array
            std::cout << twisterx::util::array_to_string(array, row - rowCount);
            if (col != col2 - 1) {
              std::cout << ",";
            }
            break;
          }
          rowCount += array->length();
        }
      }
      std::cout << "]" << std::endl;
    }
  }
  return twisterx::io::Status(Code::OK);
}

twisterx::io::Status join(const std::string &table_left,
                          const std::string &table_right,
                          int left_col_idx,
                          int right_col_idx,
                          const std::string &dest_id) {
  auto left = get_table(table_left);
  auto right = get_table(table_right);

  if (left == NULLPTR) {
    return twisterx::io::Status(Code::KeyError, "Couldn't find the left table");
  } else if (right == NULLPTR) {
    return twisterx::io::Status(Code::KeyError, "Couldn't find the right table");
  } else {
    std::shared_ptr<arrow::Table> table;
    arrow::Status status = join::join(
        left,
        right,
        left_col_idx,
        right_col_idx,
        join::JoinType::INNER,
        join::JoinAlgorithm::SORT,
        &table,
        arrow::default_memory_pool()
    );
    put_table(dest_id, table);
    return twisterx::io::Status((int) status.code(), status.message());
  }
}

int column_count(const std::string &id) {
  auto table = get_table(id);
  if (table != NULLPTR) {
    return table->num_columns();
  }
  return -1;
}

int row_count(const std::string &id) {
  auto table = get_table(id);
  if (table != NULLPTR) {
    return table->num_rows();
  }
  return -1;
}

twisterx::io::Status merge(std::vector<std::string> table_ids, const std::string &merged_tab) {
  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (auto it = table_ids.begin(); it < table_ids.end(); it++) {
    tables.push_back(get_table(*it));
  }
  arrow::Result<std::shared_ptr<arrow::Table>> result = arrow::ConcatenateTables(tables);
  if (result.status() == arrow::Status::OK()) {
    put_table(merged_tab, result.ValueOrDie());
  } else {
    return twisterx::io::Status((int) result.status().code(), result.status().message());
  }
}
}
}