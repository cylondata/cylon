/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "table_api.hpp"
#include "table_api_extended.hpp"
#include <memory>
#include <map>
#include "io/arrow_io.hpp"
#include "join/join.hpp"
#include  "util/to_string.hpp"
#include "iostream"
#include <glog/logging.h>
#include <fstream>
#include <chrono>
#include <arrow/compute/context.h>
#include <arrow/compute/api.h>
#include <future>
#include "util/arrow_utils.hpp"
#include "arrow/arrow_partition_kernels.hpp"
#include "util/uuid.hpp"
#include "arrow/arrow_all_to_all.hpp"

#include "arrow/arrow_comparator.h"

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

void RemoveTable(const std::string &id){
  table_map.erase(id);
}

twisterx::Status ReadCSV(const std::string &path,
                         const std::string &id,
                         twisterx::io::config::CSVReadOptions options) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = twisterx::io::read_csv(path, options);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> table = *result;
    LOG(INFO) << "Chunks " << table->column(0)->chunks().size();
    PutTable(id, table);
    return twisterx::Status(Code::OK, result.status().message());
  }
  return twisterx::Status(Code::IOError, result.status().message());;
}

twisterx::Status WriteCSV(const std::string &id, const std::string &path,
                          twisterx::io::config::CSVWriteOptions options) {
  auto table = GetTable(id);
  std::ofstream out_csv;
  out_csv.open(path);
  twisterx::Status status = PrintToOStream(id, 0,
                                           table->num_columns(), 0,
                                           table->num_rows(), out_csv,
                                           options.GetDelimiter(),
                                           options.IsOverrideColumnNames(),
                                           options.GetColumnNames());
  out_csv.close();
  return status;
}

twisterx::Status Print(const std::string &table_id, int col1, int col2, int row1, int row2) {
  return PrintToOStream(table_id, col1, col2, row1, row2, std::cout);
}

twisterx::Status PrintToOStream(const std::string &table_id,
                                int col1,
                                int col2,
                                int row1,
                                int row2,
                                std::ostream &out,
                                char delimiter,
                                bool use_custom_header,
                                const std::vector<std::string> &headers) {
  auto table = GetTable(table_id);
  if (table != NULLPTR) {
    // print the headers
    if (use_custom_header) {
      // check if the headers are valid
      if (headers.size() != table->num_columns()) {
        return twisterx::Status(twisterx::Code::IndexError,
                                "Provided headers doesn't match with the number of columns of the table. Given "
                                    + std::to_string(headers.size())
                                    + ", Expected " + std::to_string(table->num_columns()));
      }

      for (int col = col1; col < col2; col++) {
        out << headers[col];
        if (col != col2 - 1) {
          out << delimiter;
        } else {
          out << std::endl;
        }
      }
    }
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

              out << delimiter;
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

twisterx::Status Shuffle(twisterx::TwisterXContext *ctx,
                         const std::string &table_id,
                         const std::vector<int> &hash_columns,
                         int edge_id,
                         std::shared_ptr<arrow::Table> *table_out) {
  LOG(INFO) << "Shuffling table " << table_id << ", edge id : " << edge_id;
  auto table = GetTable(table_id);

  std::unordered_map<int, std::string> partitioned_tables{};

  // partition the tables locally
  HashPartition(table_id, hash_columns, ctx->GetWorldSize(), &partitioned_tables);

  auto neighbours = ctx->GetNeighbours(true);

  vector<std::shared_ptr<arrow::Table>> received_tables;

  // add partition of this worker to the vector
  auto partition_of_this_worker = partitioned_tables.find(ctx->GetRank());
  if (partition_of_this_worker != partitioned_tables.end()) {
    received_tables.push_back(GetTable(partition_of_this_worker->second));
  }

  // define call back to catch the receiving tables
  class AllToAllListener : public twisterx::ArrowCallback {

    vector<std::shared_ptr<arrow::Table>> tabs;

   public:
    explicit AllToAllListener(const vector<std::shared_ptr<arrow::Table>> &tabs) {
      this->tabs = tabs;
    }

    bool onReceive(int source, std::shared_ptr<arrow::Table> table) override {
      this->tabs.push_back(table);
    };
  };

  // doing all to all communication to exchange tables
  twisterx::ArrowAllToAll all_to_all(ctx, neighbours, neighbours, edge_id,
                                     std::make_shared<AllToAllListener>(received_tables),
                                     table->schema(), arrow::default_memory_pool());
  for (auto &partitioned_table : partitioned_tables) {
    if (partitioned_table.first != ctx->GetRank()) {
      all_to_all.insert(GetTable(partitioned_table.second), partitioned_table.first);
    }
  }

  LOG(INFO) << "Communicating table " << table_id;
  // now complete the communication
  all_to_all.finish();
  while (!all_to_all.isComplete()) {}
  all_to_all.close();

  LOG(INFO) << "Done communicating table " << table_id << "  received " << received_tables.size();

  // now we have the final set of tables
  arrow::Result<std::shared_ptr<arrow::Table>> concat_tables = arrow::ConcatenateTables(received_tables);

  LOG(INFO) << "Done concating tables " << table_id;

  if (concat_tables.ok()) {
    auto final_table = concat_tables.ValueOrDie();
    auto status = final_table->CombineChunks(arrow::default_memory_pool(), table_out);
    return twisterx::Status((int) status.code(), status.message());
  } else {
    return twisterx::Status((int) concat_tables.status().code(), concat_tables.status().message());
  }
}

twisterx::Status ShuffleTwoTables(twisterx::TwisterXContext *ctx,
                                  const std::string &left_table_id,
                                  const std::vector<int> &left_hash_columns,
                                  const std::string &right_table_id,
                                  const std::vector<int> &right_hash_columns,
                                  std::shared_ptr<arrow::Table> *left_table_out,
                                  std::shared_ptr<arrow::Table> *right_table_out) {
  LOG(INFO) << "Shuffling two tables";
  auto status = Shuffle(ctx, left_table_id, left_hash_columns, 0, left_table_out);
  if (status.is_ok()) {
    return Shuffle(ctx, right_table_id, right_hash_columns, 1, right_table_out);
  }
  return status;
}

twisterx::Status DistributedJoinTables(twisterx::TwisterXContext *ctx,
                                       const std::string &table_left,
                                       const std::string &table_right,
                                       twisterx::join::config::JoinConfig join_config,
                                       const std::string &dest_id) {
  // extract the tables out
  auto left = GetTable(table_left);
  auto right = GetTable(table_right);

  std::vector<int> left_hash_columns;
  left_hash_columns.push_back(join_config.GetLeftColumnIdx());

  std::vector<int> right_hash_columns;
  right_hash_columns.push_back(join_config.GetRightColumnIdx());

  std::shared_ptr<arrow::Table> left_final_table;
  std::shared_ptr<arrow::Table> right_final_table;

  auto shuffle_status = ShuffleTwoTables(ctx,
                                         table_left,
                                         left_hash_columns,
                                         table_right,
                                         right_hash_columns,
                                         &left_final_table,
                                         &right_final_table);

  if (shuffle_status.is_ok()) {
    // now do the local join
    std::shared_ptr<arrow::Table> table;
    arrow::Status status = join::joinTables(
        left_final_table,
        right_final_table,
        join_config,
        &table,
        arrow::default_memory_pool()
    );
    PutTable(dest_id, table);
    return twisterx::Status((int) status.code(), status.message());
  } else {
    return shuffle_status;
  }
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

twisterx::Status Union(const std::string &table_left, const std::string &table_right, const std::string &dest_id) {
  auto ltab = GetTable(table_left);
  auto rtab = GetTable(table_right);

  std::shared_ptr<arrow::Table> tables[2] = {ltab, rtab};

  // manual field check. todo check why  ltab->schema()->Equals(rtab->schema(), false) doesn't work

  if (ltab->num_columns() != rtab->num_columns()) {
    return twisterx::Status(twisterx::Invalid, "The no of columns of two tables are not similar. Can't perform union.");
  }

  for (int fd = 0; fd < ltab->num_columns(); ++fd) {
    if (!ltab->field(fd)->type()->Equals(rtab->field(fd)->type())) {
      return twisterx::Status(twisterx::Invalid, "The fields of two tables are not similar. Can't perform union.");
    }
  }

  int64_t eq_calls = 0, hash_calls = 0;

  class RowComparator {
   private:
    const std::shared_ptr<arrow::Table> *tables;
    twisterx::TableRowComparator *comparator;
    twisterx::RowHashingKernel *row_hashing_kernel;
    int64_t *eq, *hs;
   public:
    RowComparator(const std::shared_ptr<arrow::Table> *tables, int64_t *eq, int64_t *hs) {
      this->tables = tables;
      this->comparator = new twisterx::TableRowComparator(tables[0]->fields());
      this->row_hashing_kernel = new twisterx::RowHashingKernel(tables[0]->fields(), arrow::default_memory_pool());
      this->eq = eq;
      this->hs = hs;
    }

    // equality
    bool operator()(const std::pair<int8_t, int64_t> &record1, const std::pair<int8_t, int64_t> &record2) const {
      (*this->eq)++;
      return this->comparator->compare(
          this->tables[record1.first],
          record1.second,
          this->tables[record2.first],
          record2.second
      ) == 0;
    }

    // hashing
    size_t operator()(const std::pair<int8_t, int64_t> &record) const {
      (*this->hs)++;
      size_t hash = this->row_hashing_kernel->Hash(this->tables[record.first], record.second);
      return hash;
    }
  };

  auto row_comp = RowComparator(tables, &eq_calls, &hash_calls);

  auto buckets_pre_alloc = (ltab->num_rows() + rtab->num_rows());
  LOG(INFO) << "Buckets : " << buckets_pre_alloc;
  std::unordered_set<std::pair<int8_t, int64_t>, RowComparator, RowComparator> rows_set(
      buckets_pre_alloc,
      row_comp, row_comp
  );

  auto t1 = std::chrono::steady_clock::now();

  int64_t max = std::max(ltab->num_rows(), rtab->num_rows());
  for (int row = 0; row < max; ++row) {
    if (row < ltab->num_rows()) {
      rows_set.insert(std::make_pair<int8_t, int64_t>(0, row));
    }

    if (row < rtab->num_rows()) {
      rows_set.insert(std::make_pair<int8_t, int64_t>(1, row));
    }

    if (row % 100000 == 0) {
      LOG(INFO) << "Done " << (row + 1) * 100 / max << "%" << " N : " << row << ", Eq : " << eq_calls << ", Hs : "
                << hash_calls;
    }
  }

  auto t2 = std::chrono::steady_clock::now();

  LOG(INFO) << "Adding to Set took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms";

  std::vector<int64_t> indices_from_tabs[2];

  for (auto const &pr:rows_set) {
    indices_from_tabs[pr.first].push_back(pr.second);
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> final_data_arrays;

  t1 = std::chrono::steady_clock::now();
  // prepare final arrays
  for (int32_t c = 0; c < ltab->num_columns(); c++) {
    arrow::ArrayVector array_vector;
    for (int tab_idx = 0; tab_idx < 2; tab_idx++) {
      std::shared_ptr<arrow::Array> destination_col_array;
      arrow::Status
          status =
          twisterx::util::copy_array_by_indices(std::make_shared<std::vector<int64_t>>(indices_from_tabs[tab_idx]),
                                                tables[tab_idx]->column(c)->chunk(0),
                                                &destination_col_array,
                                                arrow::default_memory_pool());
      if (status != arrow::Status::OK()) {
        LOG(FATAL) << "Failed while copying a column to the final table from tables." << status.ToString();
        return twisterx::Status((int) status.code(), status.message());
      }
      array_vector.push_back(destination_col_array);
    }
    final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }
  t2 = std::chrono::steady_clock::now();

  LOG(INFO) << "Final array preparation took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
            << "ms";


  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(ltab->schema(), final_data_arrays);
  PutTable(dest_id, table);
  return twisterx::Status::OK();
}

twisterx::Status DistributedUnion(twisterx::TwisterXContext *ctx,
                                  const std::string &table_left,
                                  const std::string &table_right,
                                  const std::string &dest_id) {
  // extract the tables out
  auto left = GetTable(table_left);
  auto right = GetTable(table_right);

  if (left->num_columns() != right->num_columns()) {
    return twisterx::Status(twisterx::Invalid, "The no of columns of two tables are not similar. Can't perform union.");
  }

  for (int fd = 0; fd < left->num_columns(); ++fd) {
    if (!left->field(fd)->type()->Equals(right->field(fd)->type())) {
      return twisterx::Status(twisterx::Invalid, "The fields of two tables are not similar. Can't perform union.");
    }
  }

  std::vector<int32_t> hash_columns;
  for (int kI = 0; kI < left->num_columns(); ++kI) {
    hash_columns.push_back(kI);
  }

  std::shared_ptr<arrow::Table> left_final_table;
  std::shared_ptr<arrow::Table> right_final_table;

  auto shuffle_status = ShuffleTwoTables(ctx,
                                         table_left,
                                         hash_columns,
                                         table_right,
                                         hash_columns,
                                         &left_final_table,
                                         &right_final_table);

  if (shuffle_status.is_ok()) {
    auto ltab_id = PutTable(left_final_table);
    auto rtab_id = PutTable(right_final_table);

    //todo remove temp tables

    // now do the local union
    std::shared_ptr<arrow::Table> table;
    return Union(ltab_id, rtab_id, dest_id);
  } else {
    return shuffle_status;
  }

}

Status Select(const std::string &id, const std::function<bool(twisterx::Row)> &selector, const std::string &out) {

  auto src_table = GetTable(id);

  // boolean builder to hold the mask
  arrow::BooleanBuilder boolean_builder(arrow::default_memory_pool());

  for (int64_t row_index = 0; row_index < src_table->num_rows(); row_index++) {
    auto row = twisterx::Row(id, row_index);
    arrow::Status status = boolean_builder.Append(selector(row));
    if (!status.ok()) {
      return twisterx::Status(UnknownError, status.message());
    }
  }

  // building the mask
  std::shared_ptr<arrow::Array> mask;
  arrow::Status status = boolean_builder.Finish(&mask);

  if (!status.ok()) {
    return twisterx::Status(UnknownError, status.message());
  }

  std::shared_ptr<arrow::Table> out_table;
  arrow::compute::FunctionContext ctx;
  status = arrow::compute::Filter(&ctx, *src_table, *mask, &out_table);
  if (!status.ok()) {
    return twisterx::Status(UnknownError, status.message());
  }

  PutTable(out, out_table);
  return twisterx::Status::OK();
}
}