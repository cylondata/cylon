#include "table_api.hpp"
#include <memory>
#include <arrow/api.h>
#include <map>
#include "io/arrow_io.hpp"
#include "join/join.hpp"
#include  "util/to_string.hpp"
#include "iostream"
#include <glog/logging.h>
#include "util/arrow_utils.hpp"

namespace twisterx {

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

twisterx::Status read_csv(const std::string &path, const std::string &id) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = twisterx::io::read_csv(path);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> table = *result;
    put_table(id, table);
    return twisterx::Status(Code::OK, result.status().message());
  }
  return twisterx::Status(Code::IOError, result.status().message());;
}

twisterx::Status print(const std::string &table_id, int col1, int col2, int row1, int row2) {
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
  return twisterx::Status(Code::OK);
}

twisterx::Status joinTables(const std::string &table_left,
                            const std::string &table_right,
                            int left_col_idx,
                            int right_col_idx,
                            const std::string &dest_id) {
  auto left = get_table(table_left);
  auto right = get_table(table_right);

  if (left == NULLPTR) {
    return twisterx::Status(Code::KeyError, "Couldn't find the left table");
  } else if (right == NULLPTR) {
    return twisterx::Status(Code::KeyError, "Couldn't find the right table");
  } else {
    std::shared_ptr<arrow::Table> table;
    arrow::Status status = join::joinTables(
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
    return twisterx::Status((int) status.code(), status.message());
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

twisterx::Status merge(std::vector<std::string> table_ids, const std::string &merged_tab) {
  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (auto it = table_ids.begin(); it < table_ids.end(); it++) {
    tables.push_back(get_table(*it));
  }
  arrow::Result<std::shared_ptr<arrow::Table>> result = arrow::ConcatenateTables(tables);
  if (result.status() == arrow::Status::OK()) {
    put_table(merged_tab, result.ValueOrDie());
    return twisterx::Status::OK();
  } else {
    return twisterx::Status((int) result.status().code(), result.status().message());
  }
}

twisterx::Status sortTable(const std::string& id, const std::string& sortedTableId, int columnIndex) {
  auto table = get_table(id);
  if (table == NULLPTR) {
    LOG(FATAL) << "Failed to retrieve table";
    return Status(Code::KeyError, "Couldn't find the right table");
  }
  auto col = table->column(columnIndex)->chunk(0);
  std::shared_ptr<arrow::Array> indexSorts;
  arrow::Status status = SortIndices(arrow::default_memory_pool(), col, &indexSorts);

  if (status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed when sorting table to indices. " << status.ToString();
    return twisterx::Status((int)status.code(), status.message());
  }

  std::vector<std::shared_ptr<arrow::Array>> data_arrays;
  for (auto &column : table->columns()) {
    std::shared_ptr<arrow::Array> destination_col_array;
    status = twisterx::util::copy_array_by_indices(nullptr, column->chunk(0),
                                                   &destination_col_array, arrow::default_memory_pool());
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed while copying a column to the final table from left table. " << status.ToString();
      return twisterx::Status((int)status.code(), status.message());
    }
    data_arrays.push_back(destination_col_array);
  }
  // we need to put this to a new place
  std::shared_ptr<arrow::Table> sortedTable = arrow::Table::Make(table->schema(), data_arrays);
  put_table(sortedTableId, sortedTable);
  return Status::OK();
}

}