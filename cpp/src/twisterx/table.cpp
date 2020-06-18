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

#include <arrow/table.h>
#include <fstream>
#include <memory>
#include <utility>
#include "table.hpp"
#include "table_api.hpp"
#include "table_api_extended.hpp"
#include "util/uuid.hpp"
#include "arrow/arrow_types.hpp"

namespace twisterx {

Status Table::FromCSV(const std::string &path,
                      std::shared_ptr<Table> &tableOut,
                      const twisterx::io::config::CSVReadOptions &options) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::ReadCSV(path, uuid, options);
  if (status.is_ok()) {
    tableOut = std::make_shared<Table>(uuid);
  }
  return status;
}

Status Table::FromArrowTable(const std::shared_ptr<arrow::Table> &table) {
  // first check the types
  if (!twisterx::tarrow::validateArrowTableTypes(table)) {
    LOG(FATAL) << "Types not supported";
    return Status(twisterx::Invalid, "This type not supported");
  }
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  PutTable(uuid, table);
  return Status(twisterx::OK, "Loaded Successfully");
}

Status Table::FromArrowTable(const std::shared_ptr<arrow::Table> &table, std::shared_ptr<Table> *tableOut) {
  if (!twisterx::tarrow::validateArrowTableTypes(table)) {
    LOG(FATAL) << "Types not supported";
    return Status(twisterx::Invalid, "This type not supported");
  }
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  *tableOut = std::make_shared<Table>(uuid);
  PutTable(uuid, table);
  return Status(twisterx::OK, "Loaded Successfully");
}

Status Table::WriteCSV(const std::string &path, const twisterx::io::config::CSVWriteOptions &options) {
  return twisterx::WriteCSV(this->GetID(), path, options);
}

int Table::Columns() {
  return twisterx::ColumnCount(this->GetID());
}

int64_t Table::Rows() {
  return twisterx::RowCount(this->GetID());
}

void Table::Print() {
  twisterx::Print(this->GetID(), 0, this->Columns(), 0, this->Rows());
}

void Table::Print(int row1, int row2, int col1, int col2) {
  twisterx::Print(this->GetID(), col1, col2, row1, row2);
}

Status Table::Merge(const std::vector<std::shared_ptr<twisterx::Table>> &tables, std::shared_ptr<Table> *tableOut) {
  std::vector<std::string> table_ids(tables.size());
  for (auto it = tables.begin(); it < tables.end(); it++) {
    table_ids.push_back((*it)->GetID());
  }
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::Merge(table_ids, uuid);
  if (status.is_ok()) {
    *tableOut = std::make_shared<Table>(uuid);
  }
  return status;
}

Status Table::Sort(int sort_column, std::shared_ptr<Table> *tableOut) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  Status status = twisterx::SortTable(id_, uuid, sort_column);
  if (status.is_ok()) {
    *tableOut = std::make_shared<Table>(uuid);
  }
  return status;
}

Status Table::HashPartition(const std::vector<int> &hash_columns, int no_of_partitions,
                            std::vector<std::shared_ptr<twisterx::Table>> *out) {
  std::unordered_map<int, std::string> tables;
  Status status = twisterx::HashPartition(id_, hash_columns, no_of_partitions, &tables);
  if (!status.is_ok()) {
    LOG(FATAL) << "Failed to partition : " << status.get_msg();
    return status;
  }

  for (const auto &t : tables) {
    std::shared_ptr<Table> tab = std::make_shared<Table>(t.second);
    out->push_back(tab);
  }
  return Status::OK();
}

Status Table::Join(const std::shared_ptr<Table> &right,
                   twisterx::join::config::JoinConfig join_config,
                   std::shared_ptr<Table> *out) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::JoinTables(
      this->GetID(),
      right->GetID(),
      join_config,
      uuid
  );
  if (status.is_ok()) {
    *out = std::make_shared<Table>(uuid);
  }
  return status;
}

Status Table::ToArrowTable(std::shared_ptr<arrow::Table> &out) {
  std::shared_ptr<arrow::Table> tab = GetTable(id_);
  out = tab;
  return Status::OK();
}

Status Table::DistributedJoin(twisterx::TwisterXContext *ctx,
                              const shared_ptr<Table> &right,
                              twisterx::join::config::JoinConfig join_config,
                              std::shared_ptr<Table> *out) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::DistributedJoinTables(ctx, this->id_, right->id_, join_config, uuid);
  if (status.is_ok()) {
    *out = std::make_shared<Table>(uuid);
  }
  return status;
}
Status Table::Union(const shared_ptr<Table> &right, std::shared_ptr<Table> &out) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::Union(this->GetID(), right->GetID(), uuid);
  if (status.is_ok()) {
    out = std::make_shared<Table>(uuid);
  }
  return status;
}
Status Table::Select(const std::function<bool(twisterx::Row)> &selector, shared_ptr<Table> &out) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::Select(this->GetID(), selector, uuid);
  if (status.is_ok()) {
    out = std::make_shared<Table>(uuid);
  }
  return status;
}
Status Table::DistributedUnion(twisterx::TwisterXContext *ctx, const shared_ptr<Table> &right, shared_ptr<Table> &out) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  LOG(INFO) << "before";
  twisterx::Status status = twisterx::DistributedUnion(ctx, this->id_, right->id_, uuid);
  LOG(INFO) << "after";
  if (status.is_ok()) {
    out = std::make_shared<Table>(uuid);
  }
  return status;
}
void Table::Clear() {
  twisterx::RemoveTable(this->id_);
}
Table::~Table() {
  this->Clear();
}

Status Table::FromCSV(const vector<std::string> &paths,
                      const std::vector<std::shared_ptr<Table>> &tableOuts,
                      const io::config::CSVReadOptions &options) {
  std::vector<std::string> out_table_ids;
  out_table_ids.reserve(tableOuts.size());

  for (auto const &tab: tableOuts) {
    out_table_ids.push_back(tab->GetID());
  }

  return twisterx::ReadCSV(paths, out_table_ids, options);
}

Status Table::Project(const std::vector<int64_t> &project_columns, std::shared_ptr<Table> &out) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  auto status = twisterx::Project(this->id_, project_columns, uuid);

  if (status.is_ok()) {
    out = std::make_shared<Table>(uuid);
  }

  return status;
}
}
