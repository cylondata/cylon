#include <arrow/table.h>
#include "table.hpp"
#include "table_api.hpp"
#include "util/uuid.h"

namespace twisterx {

Status Table::FromCSV(const std::string &path, std::unique_ptr<Table> *tableOut) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::read_csv(path, uuid);
  if (status.is_ok()) {
    *tableOut = std::make_unique<Table>(uuid);
  }
  return status;
}

int Table::columns() {
  return twisterx::column_count(this->get_id());
}

int Table::rows() {
  return twisterx::row_count(this->get_id());
}

void Table::print() {
  twisterx::print(this->get_id(), 0, this->columns(), 0, this->rows());
}

void Table::print(int row1, int row2, int col1, int col2) {
  twisterx::print(this->get_id(), col1, col2, row1, row2);
}

Status Table::Merge(const std::vector<std::shared_ptr<twisterx::Table>>& tables, std::unique_ptr<Table> *tableOut) {
  std::vector<std::string> table_ids;
  for (auto it = tables.begin(); it < tables.end(); it++) {
    table_ids.push_back((*it)->get_id());
  }
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  twisterx::Status status = twisterx::merge(table_ids, uuid);
  if (status.is_ok()) {
    *tableOut = std::make_unique<Table>(uuid);
  }
  return status;
}

Status Table::Sort(int sort_column, std::unique_ptr<Table> *tableOut) {
  std::string uuid = twisterx::util::uuid::generate_uuid_v4();
  Status status = twisterx::sortTable(id_, uuid, sort_column);
  if (status.is_ok()) {
    *tableOut = std::make_unique<Table>(uuid);
  }
  return status;
}

Status Table::HashPartition(const std::vector<int>& hash_columns, int no_of_partitions,
    std::vector<std::shared_ptr<twisterx::Table>> *out) {
  std::vector<std::shared_ptr<arrow::Table>> tables;
  Status status = hashPartition(id_, hash_columns, no_of_partitions, &tables, arrow::default_memory_pool());
  if (!status.is_ok()) {
    LOG(FATAL) << "Failed to partition";
    return status;
  }

  for (const auto& t : tables) {
    std::string uuid = twisterx::util::uuid::generate_uuid_v4();
    put_table(uuid, t);
    std::shared_ptr<Table> tab = std::make_shared<Table>(uuid);
    out->push_back(tab);
  }
  return Status::OK();
}

}
