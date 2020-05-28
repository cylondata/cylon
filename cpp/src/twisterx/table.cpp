#include <arrow/table.h>
#include <fstream>
#include <memory>
#include "table.hpp"
#include "table_api.hpp"
#include "table_api_extended.hpp"
#include "util/uuid.hpp"
#include "arrow/arrow_types.hpp"

namespace twisterx {

Status Table::FromCSV(const std::string &path,
                      std::shared_ptr<Table> *tableOut,
                      const twisterx::io::config::CSVReadOptions &options) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::ReadCSV(path, uuid, options);
  if (status.is_ok()) {
    *tableOut = std::make_shared<Table>(Table(uuid));
  }
  return status;
}

Status Table::FromArrowTable(std::shared_ptr<arrow::Table> table) {
  // first check the types
  if (!twisterx::tarrow::validateArrowTableTypes(table)) {
    LOG(FATAL) << "Types not supported";
    return Status(twisterx::Invalid, "This type not supported");
  }
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  PutTable(uuid, table);
  return Status(twisterx::OK, "Loaded Successfully");
}

Status Table::FromArrowTable(std::shared_ptr<arrow::Table> table, std::shared_ptr<Table> *tableOut) {
  if (!twisterx::tarrow::validateArrowTableTypes(table)) {
    LOG(FATAL) << "Types not supported";
    return Status(twisterx::Invalid, "This type not supported");
  }
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  *tableOut = std::make_shared<Table>(Table(uuid));
  PutTable(uuid, table);
  return Status(twisterx::OK, "Loaded Successfully");
}

Status Table::WriteCSV(const std::string &path, const twisterx::io::config::CSVWriteOptions &options) {
  return twisterx::WriteCSV(this->get_id(), path, options);
}

int Table::columns() {
  return twisterx::ColumnCount(this->get_id());
}

int64_t Table::rows() {
  return twisterx::RowCount(this->get_id());
}

void Table::print() {
  twisterx::Print(this->get_id(), 0, this->columns(), 0, this->rows());
}

void Table::print(int row1, int row2, int col1, int col2) {
  twisterx::Print(this->get_id(), col1, col2, row1, row2);
}

Status Table::Merge(const std::vector<std::shared_ptr<twisterx::Table>> &tables, std::shared_ptr<Table> *tableOut) {
  std::vector<std::string> table_ids(tables.size());
  for (auto it = tables.begin(); it < tables.end(); it++) {
    table_ids.push_back((*it)->get_id());
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
      this->get_id(),
      right->get_id(),
      join_config,
      uuid
  );
  if (status.is_ok()) {
    *out = std::make_shared<Table>(Table(uuid));
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
  twisterx::Status status = twisterx::JoinDistributedTables(ctx, this->id_, right->id_, join_config, uuid);
  if (status.is_ok()) {
    *out = std::make_shared<Table>(Table(uuid));
  }
  return status;
}
Status Table::Union(const shared_ptr<Table> &right) {
  return twisterx::Union(this->get_id(), right->get_id());
}

}
