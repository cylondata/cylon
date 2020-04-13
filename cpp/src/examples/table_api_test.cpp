#include <table.hpp>
#include <status.hpp>
#include <iostream>
#include <io/csv_read_config.h>

int main(int argc, char *argv[]) {
  std::unique_ptr<twisterx::Table> table1, table2, joined;
  auto status1 = twisterx::Table::FromCSV("/tmp/csvd.csv", &table1,
                                          twisterx::io::config::CSVReadOptions().WithDelimiter('-'));
  auto status2 = twisterx::Table::FromCSV("/tmp/csv.csv", &table2);


  table1->Join(table2,
               twisterx::join::config::JoinConfig::RightJoin(0, 1),
               &joined);
  joined->print();
  joined->WriteCSV("/tmp/out.csv");
  return 0;
}


