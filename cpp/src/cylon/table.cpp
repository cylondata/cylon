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
#include <unordered_map>

#include "table.hpp"
#include "table_api.hpp"
#include "table_api_extended.hpp"
#include "util/uuid.hpp"
#include "arrow/arrow_types.hpp"

namespace cylon {

Status Table::FromCSV(cylon::CylonContext *ctx, const std::string &path,
                      std::shared_ptr<Table> &tableOut,
                      const cylon::io::config::CSVReadOptions &options) {
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status status = cylon::ReadCSV(ctx, path, uuid, options);
  if (status.is_ok()) {
    tableOut = std::make_shared<Table>(uuid, ctx);
  }
  return status;
}

Status Table::FromArrowTable(const std::shared_ptr<arrow::Table> &table) {
  // first check the types
  if (!cylon::tarrow::validateArrowTableTypes(table)) {
    LOG(FATAL) << "Types not supported";
    return Status(cylon::Invalid, "This type not supported");
  }
  std::string uuid = cylon::util::generate_uuid_v4();
  PutTable(uuid, table);
  return Status(cylon::OK, "Loaded Successfully");
}

Status Table::FromArrowTable(cylon::CylonContext *ctx,
                             const std::shared_ptr<arrow::Table> &table,
                             std::shared_ptr<Table> *tableOut) {
  if (!cylon::tarrow::validateArrowTableTypes(table)) {
    LOG(FATAL) << "Types not supported";
    return Status(cylon::Invalid, "This type not supported");
  }
  std::string uuid = cylon::util::generate_uuid_v4();
  *tableOut = std::make_shared<Table>(uuid, ctx);
  PutTable(uuid, table);
  return Status(cylon::OK, "Loaded Successfully");
}

Status Table::WriteCSV(const std::string &path, const cylon::io::config::CSVWriteOptions &options) {
  return cylon::WriteCSV(this->GetID(), path, options);
}

int Table::Columns() {
  return cylon::ColumnCount(this->GetID());
}

std::vector<std::string> Table::ColumnNames() {
  return cylon::ColumnNames(this->GetID());
}

int64_t Table::Rows() {
  return cylon::RowCount(this->GetID());
}

void Table::Print() {
  cylon::Print(this->GetID(), 0, this->Columns(), 0, this->Rows());
}

void Table::Print(int row1, int row2, int col1, int col2) {
  cylon::Print(this->GetID(), col1, col2, row1, row2);
}

Status Table::Merge(cylon::CylonContext *ctx,
                    const std::vector<std::shared_ptr<cylon::Table>> &tables,
                    shared_ptr<Table> &tableOut) {
  std::vector<std::string> table_ids(tables.size());
  for (auto it = tables.begin(); it < tables.end(); it++) {
    table_ids.push_back((*it)->GetID());
  }
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status status = cylon::Merge(ctx, table_ids, uuid);
  if (status.is_ok()) {
    tableOut = std::make_shared<Table>(uuid, ctx);
  }
  return status;
}

Status Table::Sort(int sort_column, shared_ptr<Table> &out) {
  std::string uuid = cylon::util::generate_uuid_v4();
  Status status = cylon::SortTable(this->ctx, id_, uuid, sort_column);
  if (status.is_ok()) {
    out = std::make_shared<Table>(uuid, ctx);
  }
  return status;
}

Status Table::HashPartition(const std::vector<int> &hash_columns, int no_of_partitions,
                            std::vector<std::shared_ptr<cylon::Table>> *out) {
  std::unordered_map<int, std::string> tables;
  Status status = cylon::HashPartition(ctx, id_, hash_columns, no_of_partitions, &tables);
  if (!status.is_ok()) {
    LOG(FATAL) << "Failed to partition : " << status.get_msg();
    return status;
  }

  for (const auto &t : tables) {
    std::shared_ptr<Table> tab = std::make_shared<Table>(t.second, this->ctx);
    out->push_back(tab);
  }
  return Status::OK();
}

Status Table::Join(const std::shared_ptr<Table> &right,
                   cylon::join::config::JoinConfig join_config,
                   std::shared_ptr<Table> *out) {
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status status = cylon::JoinTables(ctx,
                                           this->GetID(),
                                           right->GetID(),
                                           join_config,
                                           uuid);
  if (status.is_ok()) {
    *out = std::make_shared<Table>(uuid, this->ctx);
  }
  return status;
}

Status Table::ToArrowTable(std::shared_ptr<arrow::Table> &out) {
  std::shared_ptr<arrow::Table> tab = GetTable(id_);
  out = tab;
  return Status::OK();
}

Status Table::DistributedJoin(const shared_ptr<Table> &right,
                              cylon::join::config::JoinConfig join_config,
                              std::shared_ptr<Table> *out) {
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status status = cylon::DistributedJoinTables(this->ctx, this->id_,
      right->id_, join_config, uuid);
  if (status.is_ok()) {
    *out = std::make_shared<Table>(uuid, this->ctx);
  }
  return status;
}

Status Table::Select(const std::function<bool(cylon::Row)> &selector, shared_ptr<Table> &out) {
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status status = cylon::Select(ctx, this->GetID(), selector, uuid);
  if (status.is_ok()) {
    out = std::make_shared<Table>(uuid, this->ctx);
  }
  return status;
}

Status Table::DoSetOperation(SetOperation operation, const shared_ptr<Table> &right,
                             shared_ptr<Table> &out) {
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status status = operation(this->ctx, this->id_, right->id_, uuid);
  if (status.is_ok()) {
    out = std::make_shared<Table>(uuid, this->ctx);
  }
  return status;
}

Status Table::Union(const shared_ptr<Table> &right, std::shared_ptr<Table> &out) {
  return DoSetOperation(&cylon::Union, right, out);
}

Status Table::Subtract(const shared_ptr<Table> &right, std::shared_ptr<Table> &out) {
  return DoSetOperation(&cylon::Subtract, right, out);
}

Status Table::Intersect(const shared_ptr<Table> &right, std::shared_ptr<Table> &out) {
  return DoSetOperation(&cylon::Intersect, right, out);
}

Status Table::DistributedUnion(const shared_ptr<Table> &right, shared_ptr<Table> &out) {
  return DoSetOperation(&cylon::DistributedUnion, right, out);
}

Status Table::DistributedSubtract(const shared_ptr<Table> &right, shared_ptr<Table> &out) {
  return DoSetOperation(&cylon::DistributedSubtract, right, out);
}

Status Table::DistributedIntersect(const shared_ptr<Table> &right, shared_ptr<Table> &out) {
  return DoSetOperation(&cylon::DistributedIntersect, right, out);
}

void Table::Clear() {
  cylon::RemoveTable(this->id_);
}

Table::~Table() {
  this->Clear();
}

Status Table::FromCSV(cylon::CylonContext *ctx, const vector<std::string> &paths,
                      const std::vector<std::shared_ptr<Table> *> &tableOuts,
                      const io::config::CSVReadOptions &options) {
  std::vector<std::string> out_table_ids;
  out_table_ids.reserve(tableOuts.size());

  for (size_t i = 0; i < tableOuts.size(); i++) {
    out_table_ids.push_back(cylon::util::generate_uuid_v4());
  }

  auto status = cylon::ReadCSV(ctx, paths, out_table_ids, options);

  if (status.is_ok()) {
    for (size_t i = 0; i < tableOuts.size(); i++) {
      *tableOuts[i] = std::make_shared<Table>(out_table_ids[i], ctx);
    }
  }
  return status;
}

Status Table::Project(const std::vector<int64_t> &project_columns, std::shared_ptr<Table> &out) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto status = cylon::Project(this->id_, project_columns, uuid);

  if (status.is_ok()) {
    out = std::make_shared<Table>(uuid, this->ctx);
  }

  return status;
}

cylon::CylonContext *Table::GetContext() {
  return this->ctx;
}
}  // namespace cylon
