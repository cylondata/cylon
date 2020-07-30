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

#include "python/table_cython.h"

#include <arrow/api.h>
#include <arrow/python/serialize.h>

#include <fstream>
#include <map>
#include <memory>
#include <vector>
#include <utility>

#include "table_api.hpp"
#include "table_api_extended.hpp"
#include "util/uuid.hpp"
#include "join/join_config.hpp"

namespace cylon {
namespace python {
namespace table {

std::map<std::string, cylon::python::cylon_context_wrap *> context_map{};

CxTable::CxTable(std::string id) {
  id_ = id;
}

std::string CxTable::get_id() {
  return this->id_;
}

int CxTable::columns() {
  return ColumnCount(this->get_id());
}

int CxTable::rows() {
  return RowCount(this->get_id());
}

void CxTable::clear() {
}

void CxTable::show() {
  Print(this->get_id(), 0, this->columns(), 0, this->rows());
}

void CxTable::show(int row1, int row2, int col1, int col2) {
  Print(this->get_id(), col1, col2, row1, row2);
}

Status CxTable::from_csv(const std::string &path,
                         const char &delimiter,
                         const std::string &uuid) {
  auto ctx_wrap = get_new_context();
  cylon::Status status = ReadCSV(ctx_wrap->getInstance(), path, uuid,
                            cylon::io::config::CSVReadOptions().WithDelimiter(delimiter).UseThreads(
                            false).BlockSize(1 << 30));
  return status;
}

Status CxTable::to_csv(const std::string &path) {
  std::ofstream out_csv;
  out_csv.open(path);
  Status status = PrintToOStream(this->get_id(), 0,
                                 this->columns(), 0,
                                 this->rows(), out_csv);
  out_csv.close();
  return status;
}

std::string CxTable::from_pyarrow_table(std::shared_ptr<arrow::Table> table) {
  std::string uuid = cylon::util::generate_uuid_v4();
  PutTable(uuid, table);
  return uuid;
}

std::shared_ptr<arrow::Table> CxTable::to_pyarrow_table(const std::string &table_id) {
  shared_ptr<arrow::Table> table1 = GetTable(table_id);
  return table1;
}

std::string CxTable::join(const std::string &table_id,
                          cylon::join::config::JoinType type,
                          cylon::join::config::JoinAlgorithm algorithm,
                          int left_column_index,
                          int right_column_index) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto ctx_wrap = this->get_new_context();
  JoinConfig jc(type, left_column_index, right_column_index, algorithm);
  cylon::Status status = cylon::JoinTables(
      ctx_wrap->getInstance(),
      this->get_id(),
      table_id,
      jc,
      uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::join(const std::string &table_id, JoinConfig join_config) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto ctx_wrap = this->get_new_context();
  cylon::Status status = cylon::JoinTables(
      ctx_wrap->getInstance(),
      this->get_id(),
      table_id,
      join_config,
      uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::distributed_join(const std::string &table_id,
                                      cylon::join::config::JoinConfig join_config) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto ctx_wrap = this->get_new_context();
  std::cout << "distributed join , Rank  " << ctx_wrap->GetRank() << " , Size "
            << ctx_wrap->GetWorldSize() << std::endl;
  cylon::Status status =
      cylon::DistributedJoinTables(ctx_wrap->getInstance(), this->id_, table_id, join_config, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::distributed_join(cylon_context_wrap *ctx_wrap,
                                      std::string &table_id,
                                      cylon::join::config::JoinConfig join_config) {
  std::string uuid = cylon::util::generate_uuid_v4();
  std::cout << "distributed join , Rank  " << ctx_wrap->GetRank() << " , Size "
            << ctx_wrap->GetWorldSize() << std::endl;
  cylon::Status status =
      cylon::DistributedJoinTables(ctx_wrap->getInstance(), this->id_, table_id, join_config, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::distributed_join(const std::string &table_id,
                                      cylon::join::config::JoinType type,
                                      cylon::join::config::JoinAlgorithm algorithm,
                                      int left_column_index,
                                      int right_column_index) {
  cylon::join::config::JoinConfig jc(type, left_column_index, right_column_index, algorithm);
  std::string uuid = cylon::util::generate_uuid_v4();
  auto ctx_wrap = this->get_new_context();
  cylon::Status
      status = cylon::DistributedJoinTables(ctx_wrap->getInstance(), this->id_, table_id, jc, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::Union(const std::string &table_right) {
  auto ctx_wrap = this->get_new_context();
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status status = cylon::Union(ctx_wrap->getInstance(), this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::DistributedUnion(const std::string &table_right) {
  auto ctx_wrap = this->get_new_context();
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status
      status = cylon::DistributedUnion(ctx_wrap->getInstance(), this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::Intersect(const std::string &table_right) {
  auto ctx_wrap = this->get_new_context();
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status status = cylon::Intersect(ctx_wrap->getInstance(), this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::DistributedIntersect(const std::string &table_right) {
  auto ctx_wrap = this->get_new_context();
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status
      status = cylon::DistributedIntersect(ctx_wrap->getInstance(), this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::Subtract(const std::string &table_right) {
  auto ctx_wrap = this->get_new_context();
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status status = cylon::Subtract(ctx_wrap->getInstance(), this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::DistributedSubtract(const std::string &table_right) {
  auto ctx_wrap = this->get_new_context();
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status
      status = cylon::DistributedSubtract(ctx_wrap->getInstance(), this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::Project(const std::vector<int64_t> &project_columns) {
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status status = cylon::Project(this->id_, project_columns, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

// FIXME: This context handling must be replaced with a better logic
// Cython Reference passing needs to be added
cylon::python::cylon_context_wrap *CxTable::get_new_context() {
  if (context_map.size() == 0) {
    std::cout << "Creating a New Context " << std::endl;
    std::string mpi_config = "mpi";
    cylon::python::cylon_context_wrap *ctx_wrap = new cylon::python::cylon_context_wrap(mpi_config);
    std::pair<std::string, cylon::python::cylon_context_wrap *> pair("dist_context", ctx_wrap);
    context_map.insert(pair);
    return ctx_wrap;
  }
  if (context_map.size() == 1) {
    auto itr = context_map.find("dist_context");
    if (itr != context_map.end()) {
      return itr->second;
    }
  }
  return nullptr;
}

}  // namespace table
}  // namespace python
}  // namespace cylon


