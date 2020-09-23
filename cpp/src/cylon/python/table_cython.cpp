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

Status CxTable::from_csv(cylon_context_wrap *ctx_wrap,
						 const std::string &path,
                         const char &delimiter,
                         const std::string &uuid) {
  auto ctx = ctx_wrap->getInstance();
  cylon::Status status = ReadCSV(ctx, path, uuid,
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

std::string CxTable::from_pyarrow_table(cylon_context_wrap *ctx_wrap, std::shared_ptr<arrow::Table> table) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto ctx = ctx_wrap->getInstance();
  std::shared_ptr<cylon::Table> tab = std::make_shared<cylon::Table>(table, ctx);
  PutTable(uuid, tab);
  return uuid;
}

std::shared_ptr<arrow::Table> CxTable::to_pyarrow_table(const std::string &table_id) {
  std::shared_ptr<cylon::Table> table1 = GetTable(table_id);
  return table1->get_table();
}

std::string CxTable::join(cylon_context_wrap *ctx_wrap, const std::string &table_id, JoinConfig join_config) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto context = ctx_wrap->getInstance();
  cylon::Status status = cylon::JoinTables(
      context,
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

std::string CxTable::distributed_join(cylon_context_wrap *ctx_wrap,
                                      std::string &table_id,
                                      cylon::join::config::JoinConfig join_config) {
  std::string uuid = cylon::util::generate_uuid_v4();
  std::cout << "distributed join , Rank  " << ctx_wrap->GetRank() << " , Size "
            << ctx_wrap->GetWorldSize() << std::endl;
  auto context = ctx_wrap->getInstance();
  cylon::Status status =
      cylon::DistributedJoinTables(context, this->id_, table_id, join_config, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::Union(cylon_context_wrap *ctx_wrap, const std::string &table_right) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto context = ctx_wrap->getInstance();
  cylon::Status status = cylon::Union(context, this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::DistributedUnion(cylon_context_wrap *ctx_wrap, const std::string &table_right) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto context = ctx_wrap->getInstance();
  cylon::Status
      status = cylon::DistributedUnion(context, this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::Intersect(cylon_context_wrap *ctx_wrap, const std::string &table_right) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto context = ctx_wrap->getInstance();
  cylon::Status status = cylon::Intersect(context, this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::DistributedIntersect(cylon_context_wrap *ctx_wrap, const std::string &table_right) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto context = ctx_wrap->getInstance();
  cylon::Status
      status = cylon::DistributedIntersect(context, this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::Subtract(cylon_context_wrap *ctx_wrap, const std::string &table_right) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto context = ctx_wrap->getInstance();
  cylon::Status status = cylon::Subtract(context, this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::DistributedSubtract(cylon_context_wrap *ctx_wrap, const std::string &table_right) {
  std::string uuid = cylon::util::generate_uuid_v4();
  auto context = ctx_wrap->getInstance();
  cylon::Status
      status = cylon::DistributedSubtract(context, this->id_, table_right, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

std::string CxTable::Project(cylon_context_wrap *ctx_wrap, const std::vector<int64_t> &project_columns) {
  std::string uuid = cylon::util::generate_uuid_v4();
  cylon::Status status = cylon::Project(this->id_, project_columns, uuid);
  if (status.is_ok()) {
    return uuid;
  } else {
    return "";
  }
}

}  // namespace table
}  // namespace python
}  // namespace cylon


