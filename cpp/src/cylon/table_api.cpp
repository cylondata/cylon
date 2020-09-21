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

#include <arrow/compute/context.h>
#include <glog/logging.h>

#include <chrono>
#include <memory>
#include <map>
#include <future>
#include <unordered_map>
#include <utility>

#include "iostream"
#include "util/arrow_utils.hpp"
#include "util/uuid.hpp"

#include "table.hpp"

namespace cylon {
// todo make this un ordered
std::map<std::string, std::shared_ptr<cylon::Table>> table_map{};
std::mutex table_map_mutex;

std::shared_ptr<cylon::Table> GetTable(const std::string &id) {
  auto itr = table_map.find(id);
  if (itr != table_map.end()) {
    return itr->second;
  }
  LOG(INFO) << "Couldn't find table with ID " << id;
  return NULLPTR;
}

void PutTable(const std::string &id, const std::shared_ptr<cylon::Table> &table) {
  std::pair<std::string, std::shared_ptr<cylon::Table>> pair(id, table);
  table_map_mutex.lock();
  table_map.insert(pair);
  table_map_mutex.unlock();
}

std::string PutTable(const std::shared_ptr<cylon::Table> &table) {
  auto id = cylon::util::generate_uuid_v4();
  std::pair<std::string, std::shared_ptr<cylon::Table>> pair(id, table);
  table_map.insert(pair);
  return id;
}

void RemoveTable(const std::string &id) {
  table_map.erase(id);
}

Status ReadCSV(std::shared_ptr<cylon::CylonContext> &ctx,
               const std::string &path,
               const std::string &id,
               cylon::io::config::CSVReadOptions options) {
  std::shared_ptr<cylon::Table> table;
  cylon::Status status = cylon::Table::FromCSV(ctx, path, table, options);
  if (status.is_ok()) {
    PutTable(id, table);
  }
  return status;
}

Status ReadCSV(std::shared_ptr<cylon::CylonContext> &ctx,
               const std::vector<std::string> &paths,
               const std::vector<std::string> &ids,
               cylon::io::config::CSVReadOptions options) {
  if (paths.size() != ids.size()) {
    return Status(cylon::Invalid, "Size of paths and ids mismatch.");
  }
  std::vector<std::shared_ptr<Table> *> tableOuts;
  Status status = Table::FromCSV(ctx, paths, tableOuts, options);
  if (status.is_ok()) {
    for (size_t i = 0; i < ids.size(); i++) {
      PutTable(ids[i], *tableOuts[i]);
    }
  }
  return status;
}

Status WriteCSV(const std::string &id, const std::string &path,
                const cylon::io::config::CSVWriteOptions &options) {
  auto table = GetTable(id);
  return table->WriteCSV(path, options);
}

Status Print(const std::string &table_id, int col1, int col2, int row1, int row2) {
  return PrintToOStream(table_id, col1, col2, row1, row2, std::cout);
}

Status PrintToOStream(const std::string &table_id,
                      int col1,
                      int col2,
                      int row1,
                      int row2,
                      std::ostream &out,
                      char delimiter,
                      bool use_custom_header,
                      const std::vector<std::string> &headers) {
  auto table = GetTable(table_id);
  return table->PrintToOStream(col1, col2, row1, row2, out,
      delimiter, use_custom_header, headers);
}

Status DistributedJoinTables(std::shared_ptr<CylonContext> &ctx,
							 const std::string &table_left,
							 const std::string &table_right,
							 cylon::join::config::JoinConfig join_config,
							 const std::string &dest_id) {
  // extract the tables out
  auto left = GetTable(table_left);
  auto right = GetTable(table_right);

  std::shared_ptr<cylon::Table> out;
  cylon::Status status = cylon::Table::DistributedJoin(left,
      right, join_config, &out);
  if (status.is_ok()) {
    PutTable(dest_id, out);
  }
  return status;
}

Status JoinTables(std::shared_ptr<CylonContext> &ctx,
				  const std::string &table_left,
				  const std::string &table_right,
				  cylon::join::config::JoinConfig join_config,
				  const std::string &dest_id) {
  auto left = GetTable(table_left);
  auto right = GetTable(table_right);
  std::string uuid = cylon::util::generate_uuid_v4();
  std::shared_ptr<cylon::Table> out;
  cylon::Status status = cylon::Table::Join(left,
                                           right,
                                           join_config,
                                           &out);
  if (status.is_ok()) {
    PutTable(dest_id, out);
  }
  return status;
}

int ColumnCount(const std::string &id) {
  auto table = GetTable(id);
  if (table != NULLPTR) {
    return table->Columns();
  }
  return -1;
}

std::vector<std::string> ColumnNames(const std::string &id) {
  auto table = GetTable(id);
  if (table != NULLPTR) {
    return table->ColumnNames();
  } else {
    return {};
  }
}

int64_t RowCount(const std::string &id) {
  auto table = GetTable(id);
  if (table != NULLPTR) {
    return table->Rows();
  }
  return -1;
}

Status Merge(std::shared_ptr<cylon::CylonContext> &ctx,
             std::vector<std::string> table_ids,
             const std::string &merged_tab) {
  std::vector<std::shared_ptr<cylon::Table>> tables(table_ids.size());
  for (auto it = table_ids.begin(); it < table_ids.end(); it++) {
    tables.push_back(GetTable(*it));
  }
  std::shared_ptr<cylon::Table> out;
  cylon::Status status = cylon::Table::Merge(ctx, tables, out);
  if (status.is_ok()) {
    PutTable(merged_tab, out);
  }
  return status;
}

Status SortTable(std::shared_ptr<cylon::CylonContext> &ctx,
                 const std::string &id,
                 const std::string &sortedTableId,
                 int columnIndex) {
  auto table = GetTable(id);
  if (table == NULLPTR) {
    LOG(FATAL) << "Failed to retrieve table";
    return Status(Code::KeyError, "Couldn't find the right table");
  }

  shared_ptr<cylon::Table> out;
  Status status = table->Sort(columnIndex, out);
  if (status.is_ok()) {
    PutTable(sortedTableId, out);
  }
  return status;
}

Status HashPartition(std::shared_ptr<cylon::CylonContext> &ctx,
                     const std::string &id,
                     const std::vector<int> &hash_columns,
                     int no_of_partitions,
                     std::unordered_map<int, std::string> *out) {
  std::shared_ptr<cylon::Table> left_tab = GetTable(id);
  std::unordered_map<int, std::shared_ptr<cylon::Table>> tables;
  Status status = left_tab->HashPartition(hash_columns, no_of_partitions, &tables);
  if (!status.is_ok()) {
    LOG(FATAL) << "Failed to partition : " << status.get_msg();
    return status;
  }

  for (const auto &t : tables) {
    out->insert(std::pair<int, std::string>(t.first, PutTable(t.second)));
  }
  return Status::OK();
}

Status VerifyTableSchema(const std::shared_ptr<arrow::Table> &ltab,
    const std::shared_ptr<arrow::Table> &rtab) {
  // manual field check. todo check why  ltab->schema()->Equals(rtab->schema(), false) doesn't work
  if (ltab->num_columns() != rtab->num_columns()) {
    return Status(cylon::Invalid,
        "The no of columns of two tables are not similar. Can't perform union.");
  }
  for (int fd = 0; fd < ltab->num_columns(); ++fd) {
    if (!ltab->field(fd)->type()->Equals(rtab->field(fd)->type())) {
      return Status(cylon::Invalid,
          "The fields of two tables are not similar. Can't perform union.");
    }
  }
  return Status::OK();
}


Status Union(std::shared_ptr<CylonContext> &ctx,
			 const std::string &table_left,
			 const std::string &table_right,
			 const std::string &dest_id) {
  auto ltab = GetTable(table_left);
  auto rtab = GetTable(table_right);
  std::shared_ptr<cylon::Table> out;
  Status status = Table::Union(ltab, rtab, out);
  if (status.is_ok()) {
    PutTable(dest_id, out);
  }
  return status;
}

Status Subtract(std::shared_ptr<CylonContext> &ctx,
				const std::string &table_left,
				const std::string &table_right,
				const std::string &dest_id) {
  auto ltab = GetTable(table_left);
  auto rtab = GetTable(table_right);
  std::shared_ptr<cylon::Table> out;
  Status status = Table::Subtract(ltab, rtab, out);
  if (status.is_ok()) {
    PutTable(dest_id, out);
  }
  return status;
}

Status Intersect(std::shared_ptr<CylonContext> &ctx,
				 const std::string &table_left,
				 const std::string &table_right,
				 const std::string &dest_id) {
  auto ltab = GetTable(table_left);
  auto rtab = GetTable(table_right);
  std::shared_ptr<cylon::Table> out;
  Status status = Table::Intersect(ltab, rtab, out);
  if (status.is_ok()) {
    PutTable(dest_id, out);
  }
  return status;
}

Status DistributedUnion(std::shared_ptr<CylonContext> &ctx,
						const std::string &table_left,
						const std::string &table_right,
						const std::string &dest_id) {
  auto ltab = GetTable(table_left);
  auto rtab = GetTable(table_right);
  std::shared_ptr<cylon::Table> out;
  Status status = Table::DistributedUnion(ltab, rtab, out);
  if (status.is_ok()) {
    PutTable(dest_id, out);
  }
  return status;
}

Status DistributedSubtract(std::shared_ptr<cylon::CylonContext> &ctx,
						   const std::string &table_left,
						   const std::string &table_right,
						   const std::string &dest_id) {
  auto ltab = GetTable(table_left);
  auto rtab = GetTable(table_right);
  std::shared_ptr<cylon::Table> out;
  Status status = Table::DistributedSubtract(ltab, rtab, out);
  if (status.is_ok()) {
    PutTable(dest_id, out);
  }
  return status;
}

Status DistributedIntersect(std::shared_ptr<CylonContext> &ctx,
							const std::string &table_left,
							const std::string &table_right,
							const std::string &dest_id) {
  auto ltab = GetTable(table_left);
  auto rtab = GetTable(table_right);
  std::shared_ptr<cylon::Table> out_table;
  Status status = Table::DistributedIntersect(ltab, rtab, out_table);
  if (status.is_ok()) {
    PutTable(dest_id, out_table);
  }
  return status;
}

Status Select(std::shared_ptr<cylon::CylonContext> &ctx,
              const std::string &id,
              const std::function<bool(cylon::Row)> &selector,
              const std::string &dest_id) {
  auto src_table = GetTable(id);
  std::shared_ptr<cylon::Table> out_table;
  cylon::Status status = src_table->Select(selector, out_table);
  if (status.is_ok()) {
    PutTable(dest_id, out_table);
  }
  return status;
}

Status Project(const std::string &id, const std::vector<int64_t> &project_columns,
    const std::string &dest_id) {
  auto table = GetTable(id);
  std::shared_ptr<cylon::Table> out_table;
  auto status = table->Project(project_columns, out_table);
  if (status.is_ok()) {
    PutTable(dest_id, out_table);
  }
  return status;
}
}  // namespace cylon
