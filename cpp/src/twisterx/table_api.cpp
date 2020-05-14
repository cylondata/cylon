#include "table_api.hpp"
#include "table_api_extended.hpp"
#include <memory>
#include <arrow/api.h>
#include <map>
#include "io/arrow_io.hpp"
#include "join/join.hpp"
#include  "util/to_string.hpp"
#include "iostream"
#include <glog/logging.h>
#include "util/arrow_utils.hpp"
#include "arrow/arrow_partition_kernels.hpp"
#include "util/uuid.hpp"
#include "arrow/arrow_all_to_all.hpp"

namespace twisterx {

std::map<std::string, std::shared_ptr<arrow::Table>> table_map{}; //todo make this un ordered

std::shared_ptr<arrow::Table> GetTable(const std::string &id) {
  auto itr = table_map.find(id);
  if (itr != table_map.end()) {
    return itr->second;
  }
  return NULLPTR;
}

void PutTable(const std::string &id, const std::shared_ptr<arrow::Table> &table) {
  std::pair<std::string, std::shared_ptr<arrow::Table>> pair(id, table);
  table_map.insert(pair);
}

std::string PutTable(const std::shared_ptr<arrow::Table> &table) {
  auto id = twisterx::util::uuid::generate_uuid_v4();
  std::pair<std::string, std::shared_ptr<arrow::Table>> pair(id, table);
  table_map.insert(pair);
  return id;
}

twisterx::Status ReadCSV(const std::string &path,
                         const std::string &id,
                         twisterx::io::config::CSVReadOptions options) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = twisterx::io::read_csv(path, options);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> table = *result;
    PutTable(id, table);
    return twisterx::Status(Code::OK, result.status().message());
  }
  return twisterx::Status(Code::IOError, result.status().message());;
}

twisterx::Status Print(const std::string &table_id, int col1, int col2, int row1, int row2) {
  return PrintToOStream(table_id, col1, col2, row1, row2, std::cout);
}

twisterx::Status PrintToOStream(const std::string &table_id,
                                int col1,
                                int col2,
                                int row1,
                                int row2,
                                std::ostream &out) {
  auto table = GetTable(table_id);
  if (table != NULLPTR) {
    for (int row = row1; row < row2; row++) {
      for (int col = col1; col < col2; col++) {
        auto column = table->column(col);
        int rowCount = 0;
        for (int chunk = 0; chunk < column->num_chunks(); chunk++) {
          auto array = column->chunk(chunk);
          if (rowCount <= row && rowCount + array->length() > row) {
            // print this array
            out << twisterx::util::array_to_string(array, row - rowCount);
            if (col != col2 - 1) {

              out << ",";
            }
            break;
          }
          rowCount += array->length();
        }
      }
      out << std::endl;
    }
  }
  return twisterx::Status(Code::OK);
}

twisterx::Status JoinDistributedTables(
    twisterx::TwisterXContext *ctx,
    const std::string &table_left,
    const std::string &table_right,
    twisterx::join::config::JoinConfig join_config,
    const std::string &dest_id) {

  // extract the tables out
  auto left = GetTable(table_left);
  auto right = GetTable(table_right);

  std::vector<int> left_hash_columns{1};
  left_hash_columns.push_back(join_config.GetLeftColumnIdx());

  std::vector<int> right_hash_columns{1};
  right_hash_columns.push_back(join_config.GetRightColumnIdx());

  std::unordered_map<int, std::string> left_partitioned_tables{};
  std::unordered_map<int, std::string> right_partitioned_tables{};

  // partition the tables
  HashPartition(table_left, left_hash_columns, ctx->GetWorldSize(), &left_partitioned_tables);
  HashPartition(table_right, right_hash_columns, ctx->GetWorldSize(), &right_partitioned_tables);

  auto neighbours = ctx->GetNeighbours(true);

  // vectors to hold the receiving tables
  vector<std::shared_ptr<arrow::Table>> left_tables(ctx->GetWorldSize());
  vector<std::shared_ptr<arrow::Table>> right_tables(ctx->GetWorldSize());

  // add partition of this worker to the vector
  auto left_partition_of_this_worker = left_partitioned_tables.find(ctx->GetRank());
  if (left_partition_of_this_worker != left_partitioned_tables.end()) {
    left_tables.push_back(GetTable(left_partition_of_this_worker->second));
  }

  auto right_partition_of_this_worker = right_partitioned_tables.find(ctx->GetRank());
  if (right_partition_of_this_worker != right_partitioned_tables.end()) {
    right_tables.push_back(GetTable(right_partition_of_this_worker->second));
  }

  // define call back to catch the receiving tables
  class AllToAllListener : public twisterx::ArrowCallback {

    vector<std::shared_ptr<arrow::Table>> tabs;

   public:
    explicit AllToAllListener(const vector<std::shared_ptr<arrow::Table>> &tabs) {
      this->tabs = tabs;
    }

    bool onReceive(int source, std::shared_ptr<arrow::Table> table) override {
      tabs.push_back(table);
    };
  };

  // doing all to all communication to exchange tables
  twisterx::ArrowAllToAll left_all(ctx, neighbours, neighbours, 0,
                                   std::make_shared<AllToAllListener>(left_tables),
                                   left->schema(), arrow::default_memory_pool());
  for (auto & left_partitioned_table : left_partitioned_tables) {
    if (left_partitioned_table.first != ctx->GetRank()) {
      left_all.insert(GetTable(left_partitioned_table.second), left_partitioned_table.first);
    }
  }

  twisterx::ArrowAllToAll right_all(ctx, neighbours, neighbours, 0,
                                    std::make_shared<AllToAllListener>(right_tables),
                                    right->schema(), arrow::default_memory_pool());
  for (auto & right_partitioned_table : right_partitioned_tables) {
    if (right_partitioned_table.first != ctx->GetRank()) {
      right_all.insert(GetTable(right_partitioned_table.second), right_partitioned_table.first);
    }
  }

  // now complete the communication
  left_all.finish();
  right_all.finish();

  while (!left_all.isComplete()) {}
  while (!right_all.isComplete()) {}

  left_all.close();
  right_all.close();

  // now we have the final set of tables
  arrow::Result<std::shared_ptr<arrow::Table>> left_concat = arrow::ConcatenateTables(left_tables);
  arrow::Result<std::shared_ptr<arrow::Table>> right_concat = arrow::ConcatenateTables(right_tables);

  // now do the local join
  std::shared_ptr<arrow::Table> table;
  arrow::Status status = join::joinTables(
      left_concat.ValueOrDie(),
      right_concat.ValueOrDie(),
      join_config,
      &table,
      arrow::default_memory_pool()
  );
  PutTable(dest_id, table);
  return twisterx::Status((int) status.code(), status.message());
}

twisterx::Status JoinTables(const std::string &table_left,
                            const std::string &table_right,
                            twisterx::join::config::JoinConfig join_config,
                            const std::string &dest_id) {
  auto left = GetTable(table_left);
  auto right = GetTable(table_right);

  if (left == NULLPTR) {
    return twisterx::Status(Code::KeyError, "Couldn't find the left table");
  } else if (right == NULLPTR) {
    return twisterx::Status(Code::KeyError, "Couldn't find the right table");
  } else {
    std::shared_ptr<arrow::Table> table;
    arrow::Status status = join::joinTables(
        left,
        right,
        join_config,
        &table,
        arrow::default_memory_pool()
    );
    PutTable(dest_id, table);
    return twisterx::Status((int) status.code(), status.message());
  }
}

int ColumnCount(const std::string &id) {
  auto table = GetTable(id);
  if (table != NULLPTR) {
    return table->num_columns();
  }
  return -1;
}

int64_t RowCount(const std::string &id) {
  auto table = GetTable(id);
  if (table != NULLPTR) {
    return table->num_rows();
  }
  return -1;
}

twisterx::Status Merge(std::vector<std::string> table_ids, const std::string &merged_tab) {
  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (auto it = table_ids.begin(); it < table_ids.end(); it++) {
    tables.push_back(GetTable(*it));
  }
  arrow::Result<std::shared_ptr<arrow::Table>> result = arrow::ConcatenateTables(tables);
  if (result.status() == arrow::Status::OK()) {
    PutTable(merged_tab, result.ValueOrDie());
    return twisterx::Status::OK();
  } else {
    return twisterx::Status((int) result.status().code(), result.status().message());
  }
}

twisterx::Status SortTable(const std::string &id, const std::string &sortedTableId, int columnIndex) {
  auto table = GetTable(id);
  if (table == NULLPTR) {
    LOG(FATAL) << "Failed to retrieve table";
    return Status(Code::KeyError, "Couldn't find the right table");
  }
  auto col = table->column(columnIndex)->chunk(0);
  std::shared_ptr<arrow::Array> indexSorts;
  arrow::Status status = SortIndices(arrow::default_memory_pool(), col, &indexSorts);

  if (status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed when sorting table to indices. " << status.ToString();
    return twisterx::Status((int) status.code(), status.message());
  }

  std::vector<std::shared_ptr<arrow::Array>> data_arrays;
  for (auto &column : table->columns()) {
    std::shared_ptr<arrow::Array> destination_col_array;
    status = twisterx::util::copy_array_by_indices(nullptr, column->chunk(0),
                                                   &destination_col_array, arrow::default_memory_pool());
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed while copying a column to the final table from left table. " << status.ToString();
      return twisterx::Status((int) status.code(), status.message());
    }
    data_arrays.push_back(destination_col_array);
  }
  // we need to put this to a new place
  std::shared_ptr<arrow::Table> sortedTable = arrow::Table::Make(table->schema(), data_arrays);
  PutTable(sortedTableId, sortedTable);
  return Status::OK();
}

twisterx::Status HashPartition(const std::string &id, const std::vector<int> &hash_columns, int no_of_partitions,
                               std::unordered_map<int, std::string> *out) {
  std::shared_ptr<arrow::Table> left_tab = GetTable(id);
  // keep arrays for each target, these arrays are used for creating the table
  std::unordered_map<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>> data_arrays;
  std::vector<int> partitions;
  for (int t = 0; t < no_of_partitions; t++) {
    partitions.push_back(t);
    data_arrays.insert(
        std::pair<int, std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>>>(
            t, std::make_shared<std::vector<std::shared_ptr<arrow::Array>>>()));
  }

  std::vector<std::shared_ptr<arrow::Array>> arrays;
  int64_t length = 0;
  for (auto col_index: hash_columns) {
    auto column = left_tab->column(col_index);
    std::vector<int64_t> outPartitions;
    std::shared_ptr<arrow::Array> array = column->chunk(0);
    arrays.push_back(array);

    if (!(length == 0 || length == column->length())) {
      return twisterx::Status(twisterx::IndexError, "Column lengths doesnt match " + std::to_string(length));
    }
    length = column->length();
  }

  // first we partition the table
  std::vector<int64_t> outPartitions;
  twisterx::Status
      status = HashPartitionArrays(arrow::default_memory_pool(), arrays, length, partitions, &outPartitions);
  if (!status.is_ok()) {
    LOG(FATAL) << "Failed to create the hash partition";
    return status;
  }

  for (int i = 0; i < left_tab->num_columns(); i++) {
    std::shared_ptr<arrow::DataType> type = left_tab->column(i)->chunk(0)->type();
    std::shared_ptr<arrow::Array> array = left_tab->column(i)->chunk(0);

    std::shared_ptr<ArrowArraySplitKernel> splitKernel;
    status = CreateSplitter(type, arrow::default_memory_pool(), &splitKernel);
    if (!status.is_ok()) {
      LOG(FATAL) << "Failed to create the splitter";
      return status;
    }

    // this one outputs arrays for each target as a map
    std::unordered_map<int, std::shared_ptr<arrow::Array>> splited_arrays;
    splitKernel->Split(array, outPartitions, partitions, splited_arrays);

    for (const auto &x : splited_arrays) {
      std::shared_ptr<std::vector<std::shared_ptr<arrow::Array>>> cols = data_arrays[x.first];
      cols->push_back(x.second);
    }
  }
  // now insert these array to
  for (const auto &x : data_arrays) {
    std::shared_ptr<arrow::Table> table = arrow::Table::Make(left_tab->schema(), *x.second);
    out->insert(std::pair<int, std::string>(x.first, PutTable(table)));
  }
  return twisterx::Status::OK();
}

}