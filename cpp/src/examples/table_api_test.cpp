#include <table.hpp>
#include <status.hpp>
#include <iostream>

int main(int argc, char *argv[]) {
  std::unique_ptr<twisterx::Table> table1, table2, joined;
  auto status1 = twisterx::Table::FromCSV("/tmp/csv.csv", &table1);
  auto status2 = twisterx::Table::FromCSV("/tmp/csv.csv", &table2);

  table1->Join(table2,
               twisterx::join::config::JoinConfig::InnerJoin(0, 0),
               &joined);
  joined->print();
  joined->WriteCSV("/tmp/out.csv");
  return 0;
}


