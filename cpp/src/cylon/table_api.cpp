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
#include "ctx/arrow_memory_pool_utils.h"

namespace cylon {

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
  auto id = cylon::util::uuid::generate_uuid_v4();
  std::pair<std::string, std::shared_ptr<arrow::Table>> pair(id, table);
  table_map.insert(pair);
  return id;
}

void RemoveTable(const std::string &id) {
  table_map.erase(id);
}

cylon::Status ReadCSV(cylon::CylonContext *ctx,
                      const std::string &path,
                      const std::string &id,
                      cylon::io::config::CSVReadOptions options) {
  arrow::Result<std::shared_ptr<arrow::Table>> result = cylon::io::read_csv(ctx, path, options);
  if (result.ok()) {
    std::shared_ptr<arrow::Table> table = *result;
    LOG(INFO) << "Chunks " << table->column(0)->chunks().size();
    if (table->column(0)->chunks().size() > 1) {
      auto status = table->CombineChunks(ToArrowPool(ctx), &table);
      if (!status.ok()) {
        return cylon::Status(Code::IOError, status.message());;
      }
    }
    PutTable(id, table);
    return cylon::Status(Code::OK, result.status().message());
  }
  return cylon::Status(Code::IOError, result.status().message());;
}

void ReadCSVThread(cylon::CylonContext *ctx, const std::string &path,
                   const std::string &id,
                   cylon::io::config::CSVReadOptions options,
                   const std::shared_ptr<std::promise<cylon::Status>> &status_promise) {
  status_promise->set_value(ReadCSV(ctx, path, id, options));
}

cylon::Status ReadCSV(cylon::CylonContext *ctx,
                      const std::vector<std::string> &paths,
                      const std::vector<std::string> &ids,
                      cylon::io::config::CSVReadOptions options) {

  if (paths.size() != ids.size()) {
    return cylon::Status(cylon::Invalid, "Size of paths and ids mismatch.");
  }

  if (options.IsConcurrentFileReads()) {
    std::vector<std::pair<std::future<cylon::Status>, std::thread>> futures;
    futures.reserve(paths.size());
    for (uint64_t kI = 0; kI < paths.size(); ++kI) {
      auto read_promise = std::make_shared<std::promise<cylon::Status>>();
      futures.push_back(std::pair<std::future<cylon::Status>, std::thread>
                            (read_promise->get_future(),
                             std::thread(ReadCSVThread,
                                         ctx,
                                         paths[kI],
                                         ids[kI],
                                         options,
                                         read_promise)));
    }
    bool all_passed = false;
    for (auto &future: futures) {
      all_passed &= future.first.get().is_ok();
      future.second.join();
    }
    return all_passed ? cylon::Status::OK() : cylon::Status(cylon::IOError, "Failed to read the csv files");
  } else {
    auto status = cylon::Status::OK();
    for (int kI = 0; kI < paths.size(); ++kI) {
      status = ReadCSV(ctx, paths[kI], ids[kI], options);
      if (!status.is_ok()) {
        return status;
      }
    }
    return status;
  }
}

cylon::Status WriteCSV(const std::string &id, const std::string &path,
                       const cylon::io::config::CSVWriteOptions &options) {
  auto table = GetTable(id);
  std::ofstream out_csv;
  out_csv.open(path);
  cylon::Status status = PrintToOStream(id, 0,
                                        table->num_columns(), 0,
                                        table->num_rows(), out_csv,
                                        options.GetDelimiter(),
                                        options.IsOverrideColumnNames(),
                                        options.GetColumnNames());
  out_csv.close();
  return status;
}

cylon::Status Print(const std::string &table_id, int col1, int col2, int row1, int row2) {
  return PrintToOStream(table_id, col1, col2, row1, row2, std::cout);
}

cylon::Status PrintToOStream(const std::string &table_id,
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
        return cylon::Status(cylon::Code::IndexError,
                             "Provided headers doesn't match with the number of columns of the table. Given "
                                 + std::to_string(headers.size()) + ", Expected "
                                 + std::to_string(table->num_columns()));
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
            out << cylon::util::array_to_string(array, row - rowCount);
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
  return cylon::Status(Code::OK);
}

cylon::Status Shuffle(cylon::CylonContext *ctx,
                      const std::string &table_id,
                      const std::vector<int> &hash_columns,
                      int edge_id,
                      std::shared_ptr<arrow::Table> *table_out) {
  auto table = GetTable(table_id);

  std::unordered_map<int, std::string> partitioned_tables{};

  // partition the tables locally
  HashPartition(ctx, table_id, hash_columns, ctx->GetWorldSize(), &partitioned_tables);

  auto neighbours = ctx->GetNeighbours(true);

  vector<std::shared_ptr<arrow::Table>> received_tables;

  // define call back to catch the receiving tables
  class AllToAllListener : public cylon::ArrowCallback {

    vector<std::shared_ptr<arrow::Table>> *tabs;
    int workerId;

   public:
    explicit AllToAllListener(vector<std::shared_ptr<arrow::Table>> *tabs, int workerId) {
      this->tabs = tabs;
      this->workerId = workerId;
    }

    bool onReceive(int source, std::shared_ptr<arrow::Table> table) override {
      this->tabs->push_back(table);
      return true;
    };
  };

  // doing all to all communication to exchange tables
  cylon::ArrowAllToAll all_to_all(ctx, neighbours, neighbours, edge_id,
                                  std::make_shared<AllToAllListener>(&received_tables, ctx->GetRank()),
                                  table->schema(), cylon::ToArrowPool(ctx));
  for (auto &partitioned_table : partitioned_tables) {
    if (partitioned_table.first != ctx->GetRank()) {
      all_to_all.insert(GetTable(partitioned_table.second), partitioned_table.first);
    } else {
      received_tables.push_back(GetTable(partitioned_table.second));
    }
  }

  // now complete the communication
  all_to_all.finish();
  while (!all_to_all.isComplete()) {}
  all_to_all.close();

  // now we have the final set of tables
  LOG(INFO) << "Concatenating tables, Num of tables :  " << received_tables.size();
  arrow::Result<std::shared_ptr<arrow::Table>> concat_tables = arrow::ConcatenateTables(received_tables);

  if (concat_tables.ok()) {
    auto final_table = concat_tables.ValueOrDie();
    LOG(INFO) << "Done concatenating tables, rows :  " << final_table->num_rows();
    auto status = final_table->CombineChunks(cylon::ToArrowPool(ctx), table_out);
    return cylon::Status((int) status.code(), status.message());
  } else {
    return cylon::Status((int) concat_tables.status().code(), concat_tables.status().message());
  }
}

cylon::Status ShuffleTwoTables(cylon::CylonContext *ctx,
                               const std::string &left_table_id,
                               const std::vector<int> &left_hash_columns,
                               const std::string &right_table_id,
                               const std::vector<int> &right_hash_columns,
                               std::shared_ptr<arrow::Table> *left_table_out,
                               std::shared_ptr<arrow::Table> *right_table_out) {
  LOG(INFO) << "Shuffling two tables with total rows : "
            << GetTable(left_table_id)->num_rows() + GetTable(right_table_id)->num_rows();
  auto status = Shuffle(ctx, left_table_id, left_hash_columns, ctx->GetNextSequence(), left_table_out);
  if (status.is_ok()) {
    LOG(INFO) << "Left table shuffled";
    return Shuffle(ctx, right_table_id, right_hash_columns, ctx->GetNextSequence(), right_table_out);
  }
  return status;
}

cylon::Status DistributedJoinTables(cylon::CylonContext *ctx,
                                    const std::string &table_left,
                                    const std::string &table_right,
                                    cylon::join::config::JoinConfig join_config,
                                    const std::string &dest_id) {
  // extract the tables out
  auto left = GetTable(table_left);
  auto right = GetTable(table_right);

  // check whether the world size is 1
  if (ctx->GetWorldSize() == 1) {
    std::shared_ptr<arrow::Table> table;
    arrow::Status status = join::joinTables(
        left,
        right,
        join_config,
        &table,
        cylon::ToArrowPool(ctx)
    );
    PutTable(dest_id, table);
    return cylon::Status((int) status.code(), status.message());
  }

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
        cylon::ToArrowPool(ctx)
    );
    PutTable(dest_id, table);
    return cylon::Status((int) status.code(), status.message());
  } else {
    return shuffle_status;
  }
}

cylon::Status JoinTables(cylon::CylonContext *ctx,
                         const std::string &table_left,
                         const std::string &table_right,
                         cylon::join::config::JoinConfig join_config,
                         const std::string &dest_id) {
  auto left = GetTable(table_left);
  auto right = GetTable(table_right);

  if (left == NULLPTR) {
    return cylon::Status(Code::KeyError, "Couldn't find the left table");
  } else if (right == NULLPTR) {
    return cylon::Status(Code::KeyError, "Couldn't find the right table");
  } else {
    std::shared_ptr<arrow::Table> table;
    arrow::Status status = join::joinTables(
        left,
        right,
        join_config,
        &table,
        cylon::ToArrowPool(ctx)
    );
    PutTable(dest_id, table);
    return cylon::Status((int) status.code(), status.message());
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

cylon::Status Merge(cylon::CylonContext *ctx,
                    std::vector<std::string> table_ids,
                    const std::string &merged_tab) {
  std::vector<std::shared_ptr<arrow::Table>> tables;
  for (auto it = table_ids.begin(); it < table_ids.end(); it++) {
    tables.push_back(GetTable(*it));
  }
  arrow::Result<std::shared_ptr<arrow::Table>> result = arrow::ConcatenateTables(tables);
  if (result.status() == arrow::Status::OK()) {
    std::shared_ptr<arrow::Table> combined;
    result.ValueOrDie()->CombineChunks(cylon::ToArrowPool(ctx), &combined);
    PutTable(merged_tab, combined);
    return cylon::Status::OK();
  } else {
    return cylon::Status((int) result.status().code(), result.status().message());
  }
}

cylon::Status SortTable(cylon::CylonContext *ctx,
                        const std::string &id,
                        const std::string &sortedTableId,
                        int columnIndex) {
  auto table = GetTable(id);
  if (table == NULLPTR) {
    LOG(FATAL) << "Failed to retrieve table";
    return Status(Code::KeyError, "Couldn't find the right table");
  }
  auto col = table->column(columnIndex)->chunk(0);
  std::shared_ptr<arrow::Array> indexSorts;
  arrow::Status status = SortIndices(cylon::ToArrowPool(ctx), col, &indexSorts);

  if (status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed when sorting table to indices. " << status.ToString();
    return cylon::Status((int) status.code(), status.message());
  }

  std::vector<std::shared_ptr<arrow::Array>> data_arrays;
  for (auto &column : table->columns()) {
    std::shared_ptr<arrow::Array> destination_col_array;
    status = cylon::util::copy_array_by_indices(nullptr, column->chunk(0),
                                                &destination_col_array, cylon::ToArrowPool(ctx));
    if (status != arrow::Status::OK()) {
      LOG(FATAL) << "Failed while copying a column to the final table from left table. " << status.ToString();
      return cylon::Status((int) status.code(), status.message());
    }
    data_arrays.push_back(destination_col_array);
  }
  // we need to put this to a new place
  std::shared_ptr<arrow::Table> sortedTable = arrow::Table::Make(table->schema(), data_arrays);
  PutTable(sortedTableId, sortedTable);
  return Status::OK();
}

cylon::Status HashPartition(cylon::CylonContext *ctx,
                            const std::string &id,
                            const std::vector<int> &hash_columns,
                            int no_of_partitions,
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
      return cylon::Status(cylon::IndexError, "Column lengths doesnt match " + std::to_string(length));
    }
    length = column->length();
  }

  // first we partition the table
  std::vector<int64_t> outPartitions;
  cylon::Status
      status = HashPartitionArrays(cylon::ToArrowPool(ctx), arrays, length, partitions, &outPartitions);
  if (!status.is_ok()) {
    LOG(FATAL) << "Failed to create the hash partition";
    return status;
  }

  for (int i = 0; i < left_tab->num_columns(); i++) {
    std::shared_ptr<arrow::DataType> type = left_tab->column(i)->chunk(0)->type();
    std::shared_ptr<arrow::Array> array = left_tab->column(i)->chunk(0);

    std::shared_ptr<ArrowArraySplitKernel> splitKernel;
    status = CreateSplitter(type, cylon::ToArrowPool(ctx), &splitKernel);
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
  return cylon::Status::OK();
}

class RowComparator {
 private:
  const std::shared_ptr<arrow::Table> *tables;
  std::shared_ptr<cylon::TableRowComparator> comparator;
  std::shared_ptr<cylon::RowHashingKernel> row_hashing_kernel;
  int64_t *eq, *hs;

 public:
  RowComparator(cylon::CylonContext *ctx,
                const std::shared_ptr<arrow::Table> *tables,
                int64_t *eq,
                int64_t *hs) {
    this->tables = tables;
    this->comparator = std::make_shared<cylon::TableRowComparator>(tables[0]->fields());
    this->row_hashing_kernel = std::make_shared<cylon::RowHashingKernel>(tables[0]->fields(), cylon::ToArrowPool(ctx));
    this->eq = eq;
    this->hs = hs;
  }

  // equality
  bool operator()(const std::pair<int8_t, int64_t> &record1, const std::pair<int8_t, int64_t> &record2) const {
    (*this->eq)++;
    return this->comparator->compare(this->tables[record1.first], record1.second,
                                     this->tables[record2.first], record2.second) == 0;
  }

  // hashing
  size_t operator()(const std::pair<int8_t, int64_t> &record) const {
    (*this->hs)++;
    size_t hash = this->row_hashing_kernel->Hash(this->tables[record.first], record.second);
    return hash;
  }
};

cylon::Status VerifyTableSchema(const std::shared_ptr<arrow::Table> &ltab, const std::shared_ptr<arrow::Table> &rtab) {
  // manual field check. todo check why  ltab->schema()->Equals(rtab->schema(), false) doesn't work

  if (ltab->num_columns() != rtab->num_columns()) {
    return cylon::Status(cylon::Invalid, "The no of columns of two tables are not similar. Can't perform union.");
  }

  for (int fd = 0; fd < ltab->num_columns(); ++fd) {
    if (!ltab->field(fd)->type()->Equals(rtab->field(fd)->type())) {
      return cylon::Status(cylon::Invalid, "The fields of two tables are not similar. Can't perform union.");
    }
  }

  return cylon::Status::OK();
}
/**
 * creates an Arrow array based on col_idx, filtered by row_indices
 * @param ctx
 * @param table
 * @param col_idx
 * @param row_indices
 * @param array_vector
 * @return
 */
cylon::Status PrepareArray(cylon::CylonContext *ctx,
                           const std::shared_ptr<arrow::Table> &table,
                           const int32_t col_idx,
                           const std::shared_ptr<std::vector<int64_t>> &row_indices,
                           arrow::ArrayVector &array_vector) {
  std::shared_ptr<arrow::Array> destination_col_array;
  arrow::Status ar_status = cylon::util::copy_array_by_indices(row_indices, table->column(col_idx)->chunk(0),
                                                               &destination_col_array, cylon::ToArrowPool(ctx));
  if (ar_status != arrow::Status::OK()) {
    LOG(FATAL) << "Failed while copying a column to the final table from tables." << ar_status.ToString();
    return cylon::Status((int) ar_status.code(), ar_status.message());
  }
  array_vector.push_back(destination_col_array);

  return cylon::Status::OK();
}

cylon::Status Union(cylon::CylonContext *ctx,
                    const std::string &table_left,
                    const std::string &table_right,
                    const std::string &dest_id) {
  auto ltab = GetTable(table_left);
  auto rtab = GetTable(table_right);

  cylon::Status status = VerifyTableSchema(ltab, rtab);
  if (!status.is_ok()) return status;

  std::shared_ptr<arrow::Table> tables[2] = {ltab, rtab};

  int64_t eq_calls = 0, hash_calls = 0;

  auto row_comp = RowComparator(ctx, tables, &eq_calls, &hash_calls);

  auto buckets_pre_alloc = (ltab->num_rows() + rtab->num_rows());
  LOG(INFO) << "Buckets : " << buckets_pre_alloc;
  std::unordered_set<std::pair<int8_t, int64_t>, RowComparator, RowComparator>
      rows_set(buckets_pre_alloc, row_comp, row_comp);

  auto t1 = std::chrono::steady_clock::now();

  const int64_t max = std::max(ltab->num_rows(), rtab->num_rows());
  const int8_t table0 = 0;
  const int8_t table1 = 1;
  const int64_t print_threshold = max / 10;
  for (int64_t row = 0; row < max; ++row) {
    if (row < ltab->num_rows()) {
      rows_set.insert(std::pair<int8_t, int64_t>(table0, row));
    }

    if (row < rtab->num_rows()) {
      rows_set.insert(std::pair<int8_t, int64_t>(table1, row));
    }

    if (row % print_threshold == 0) {
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
  for (int32_t col_idx = 0; col_idx < ltab->num_columns(); col_idx++) {
    arrow::ArrayVector array_vector;
    for (int tab_idx = 0; tab_idx < 2; tab_idx++) {
      status = PrepareArray(ctx,
                            tables[tab_idx],
                            col_idx,
                            std::make_shared<std::vector<int64_t>>(indices_from_tabs[tab_idx]),
                            array_vector);

      if (!status.is_ok()) return status;
    }
    final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }
  t2 = std::chrono::steady_clock::now();

  LOG(INFO) << "Final array preparation took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
            << "ms";


  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(ltab->schema(), final_data_arrays);
  PutTable(dest_id, table);
  return cylon::Status::OK();
}

cylon::Status Subtract(cylon::CylonContext *ctx,
                       const std::string &table_left,
                       const std::string &table_right,
                       const std::string &dest_id) {
  auto ltab = GetTable(table_left);
  auto rtab = GetTable(table_right);

  cylon::Status status = VerifyTableSchema(ltab, rtab);
  if (!status.is_ok()) return status;

  std::shared_ptr<arrow::Table> tables[2] = {ltab, rtab};

  int64_t eq_calls = 0, hash_calls = 0;

  auto row_comp = RowComparator(ctx, tables, &eq_calls, &hash_calls);

  auto buckets_pre_alloc = ltab->num_rows();
  LOG(INFO) << "Buckets : " << buckets_pre_alloc;
  std::unordered_set<std::pair<int8_t, int64_t>, RowComparator, RowComparator>
      left_row_set(buckets_pre_alloc, row_comp, row_comp);

  auto t1 = std::chrono::steady_clock::now();

  // first populate left table in the hash set
  int64_t print_offset = ltab->num_rows() / 4, next_print = print_offset;
  for (int64_t row = 0; row < ltab->num_rows(); ++row) {
    left_row_set.insert(std::pair<int8_t, int64_t>(0, row));

    if (row == next_print) {
      LOG(INFO) << "Done " << (row + 1) * 100 / ltab->num_rows() << "%" << " N : " << row << ", Eq : " << eq_calls
                << ", Hs : " << hash_calls;
      next_print += print_offset;
    }
  }

  // then remove matching rows using hash map comparator
  print_offset = rtab->num_rows() / 4;
  next_print = print_offset;
  for (int64_t row = 0; row < rtab->num_rows(); ++row) {
    left_row_set.erase(std::pair<int8_t, int64_t>(1, row)); // finds a matching row from left and erase it

    if (row == next_print) {
      LOG(INFO) << "Done " << (row + 1) * 100 / rtab->num_rows() << "%" << " N : " << row << ", Eq : " << eq_calls
                << ", Hs : " << hash_calls;
      next_print += print_offset;
    }
  }

  auto t2 = std::chrono::steady_clock::now();

  LOG(INFO) << "Adding to Set took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms";

  std::vector<int64_t> left_indices;
  left_indices.reserve(left_row_set.size()); // reserve space for vec

  for (auto const &pr:left_row_set) {
    left_indices.push_back(pr.second);
  }

  std::vector<std::shared_ptr<arrow::ChunkedArray>> final_data_arrays;

  t1 = std::chrono::steady_clock::now();
  // prepare final arrays
  for (int32_t col_idx = 0; col_idx < ltab->num_columns(); col_idx++) {
    arrow::ArrayVector array_vector;

    status = PrepareArray(ctx, ltab, col_idx, std::make_shared<std::vector<int64_t>>(left_indices), array_vector);

    if (!status.is_ok()) return status;

    final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }
  t2 = std::chrono::steady_clock::now();

  LOG(INFO) << "Final array preparation took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
            << "ms";

  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(ltab->schema(), final_data_arrays);
  PutTable(dest_id, table);
  return cylon::Status::OK();
}

cylon::Status Intersect(cylon::CylonContext *ctx,
                        const std::string &table_left,
                        const std::string &table_right,
                        const std::string &dest_id) {
  auto ltab = GetTable(table_left);
  auto rtab = GetTable(table_right);

  cylon::Status status = VerifyTableSchema(ltab, rtab);
  if (!status.is_ok()) return status;

  std::shared_ptr<arrow::Table> tables[2] = {ltab, rtab};

  int64_t eq_calls = 0, hash_calls = 0;

  auto row_comp = RowComparator(ctx, tables, &eq_calls, &hash_calls);

  auto buckets_pre_alloc = (ltab->num_rows() + rtab->num_rows());
  LOG(INFO) << "Buckets : " << buckets_pre_alloc;
  std::unordered_set<std::pair<int8_t, int64_t>, RowComparator, RowComparator>
      rows_set(buckets_pre_alloc, row_comp, row_comp);

  auto t1 = std::chrono::steady_clock::now();

  // first populate left table in the hash set
  int64_t print_offset = ltab->num_rows() / 4, next_print = print_offset;
  for (int64_t row = 0; row < ltab->num_rows(); ++row) {
    rows_set.insert(std::pair<int8_t, int64_t>(0, row));

    if (row == next_print) {
      LOG(INFO) << "left done " << (row + 1) * 100 / ltab->num_rows() << "%" << " N : " << row << ", Eq : " << eq_calls
                << ", Hs : " << hash_calls;
      next_print += print_offset;
    }
  }

  std::vector<int64_t> left_indices;

  // then add matching rows to the indices_from_tabs vector
  print_offset = rtab->num_rows() / 4;
  next_print = print_offset;
  for (int64_t row = 0; row < rtab->num_rows(); ++row) {
    auto matching_row_it = rows_set.find(std::pair<int8_t, int64_t>(1, row));

    if (matching_row_it != rows_set.end()) {
      std::pair<int8_t, int64_t> matching_row = *matching_row_it;
      left_indices.push_back(matching_row.second);
    }

    if (row == next_print) {
      LOG(INFO) << "right done " << (row + 1) * 100 / rtab->num_rows() << "%" << " N : " << row << ", Eq : " << eq_calls
                << ", Hs : " << hash_calls;
      next_print += print_offset;
    }
  }

  auto t2 = std::chrono::steady_clock::now();

  LOG(INFO) << "Adding to Set took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms";

  std::vector<std::shared_ptr<arrow::ChunkedArray>> final_data_arrays;

  t1 = std::chrono::steady_clock::now();
  // prepare final arrays
  for (int32_t col_idx = 0; col_idx < ltab->num_columns(); col_idx++) {
    arrow::ArrayVector array_vector;
    status = PrepareArray(ctx, ltab, col_idx, std::make_shared<std::vector<int64_t>>(left_indices), array_vector);

    if (!status.is_ok()) return status;

    final_data_arrays.push_back(std::make_shared<arrow::ChunkedArray>(array_vector));
  }

  t2 = std::chrono::steady_clock::now();

  LOG(INFO) << "Final array preparation took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
            << "ms";

  // create final table
  std::shared_ptr<arrow::Table> table = arrow::Table::Make(ltab->schema(), final_data_arrays);
  PutTable(dest_id, table);
  return cylon::Status::OK();
}

typedef Status
(*LocalSetOperation)(CylonContext *, const std::string &, const std::string &, const std::string &);
cylon::Status DoDistributedSetOperation(CylonContext *ctx,
                                        LocalSetOperation local_operation,
                                        const std::string &table_left,
                                        const std::string &table_right,
                                        const std::string &dest_id) {
  // extract the tables out
  auto left = GetTable(table_left);
  auto right = GetTable(table_right);

  cylon::Status status = VerifyTableSchema(left, right);
  if (!status.is_ok()) return status;

  std::vector<int32_t> hash_columns;
  hash_columns.reserve(left->num_columns());
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

    // now do the local union
    std::shared_ptr<arrow::Table> table;
    status = local_operation(ctx, ltab_id, rtab_id, dest_id);

    RemoveTable(ltab_id);
    RemoveTable(rtab_id);

    return status;
  } else {
    return shuffle_status;
  }
}

cylon::Status DistributedUnion(cylon::CylonContext *ctx,
                               const std::string &table_left,
                               const std::string &table_right,
                               const std::string &dest_id) {
  return DoDistributedSetOperation(ctx, &Union, table_left, table_right, dest_id);
}

Status DistributedSubtract(cylon::CylonContext *ctx,
                           const std::string &table_left,
                           const std::string &table_right,
                           const std::string &dest_id) {
  return DoDistributedSetOperation(ctx, &Subtract, table_left, table_right, dest_id);
}

Status DistributedIntersect(cylon::CylonContext *ctx,
                            const std::string &table_left,
                            const std::string &table_right,
                            const std::string &dest_id) {
  return DoDistributedSetOperation(ctx, &Intersect, table_left, table_right, dest_id);
}

Status Select(cylon::CylonContext *ctx,
              const std::string &id,
              const std::function<bool(cylon::Row)> &selector,
              const std::string &out) {

  auto src_table = GetTable(id);

  // boolean builder to hold the mask
  arrow::BooleanBuilder boolean_builder(cylon::ToArrowPool(ctx));

  for (int64_t row_index = 0; row_index < src_table->num_rows(); row_index++) {
    auto row = cylon::Row(id, row_index);
    arrow::Status status = boolean_builder.Append(selector(row));
    if (!status.ok()) {
      return cylon::Status(UnknownError, status.message());
    }
  }

  // building the mask
  std::shared_ptr<arrow::Array> mask;
  arrow::Status status = boolean_builder.Finish(&mask);

  if (!status.ok()) {
    return cylon::Status(UnknownError, status.message());
  }

  std::shared_ptr<arrow::Table> out_table;
  arrow::compute::FunctionContext func_ctx;
  status = arrow::compute::Filter(&func_ctx, *src_table, *mask, &out_table);
  if (!status.ok()) {
    return cylon::Status(UnknownError, status.message());
  }

  PutTable(out, out_table);
  return cylon::Status::OK();
}

Status Project(const std::string &id, const std::vector<int64_t> &project_columns, const std::string &out) {
  auto table = GetTable(id);
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;
  std::vector<std::shared_ptr<arrow::ChunkedArray>> column_arrays;
  schema_vector.reserve(project_columns.size());

  for (auto const &col_index: project_columns) {
    schema_vector.push_back(table->field(col_index));
    auto chunked_array = std::make_shared<arrow::ChunkedArray>(table->column(col_index)->chunks());
    column_arrays.push_back(chunked_array);
  }

  auto schema = std::make_shared<arrow::Schema>(schema_vector);

  std::shared_ptr<arrow::Table> projected_table = arrow::Table::Make(schema, column_arrays);

  PutTable(out, projected_table);
  return cylon::Status::OK();
}
}