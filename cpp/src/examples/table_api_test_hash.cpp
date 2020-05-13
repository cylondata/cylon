#include <table.hpp>
#include <status.hpp>
#include <iostream>
#include <io/csv_read_config.h>

using namespace twisterx;
using namespace twisterx::join::config;

int main(int argc, char *argv[]) {

  std::shared_ptr<Table> table1, table2, joined;

  auto status1 = Table::FromCSV("/tmp/csv.csv", &table1,
								io::config::CSVReadOptions().WithDelimiter(','));
  auto status2 = Table::FromCSV("/tmp/csv.csv", &table2);

  const JoinConfig jc = JoinConfig::LeftJoin(0, 1, JoinAlgorithm::HASH);

  const Status &status = table1->Join(table2, jc, &joined);

  joined->print();
  joined->WriteCSV("/tmp/out.csv");

  return 0;
}


