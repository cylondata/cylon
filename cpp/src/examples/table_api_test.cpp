#include <table.hpp>
#include <status.hpp>
#include <iostream>
#include <io/csv_read_config.h>

using namespace twisterx;

int main(int argc, char *argv[]) {

  std::shared_ptr<twisterx::Table> table1, table2, joined;
  std::shared_ptr<Table> *sorted;
  auto status1 = twisterx::Table::FromCSV("/tmp/csv.csv", &table1,
										  twisterx::io::config::CSVReadOptions().WithDelimiter(','));
  auto status2 = twisterx::Table::FromCSV("/tmp/csv.csv", &table2);

  table1->Join(table2,
			   twisterx::join::config::JoinConfig::RightJoin(0, 1),
			   &joined);
  joined->print();
  joined->WriteCSV("/tmp/out.csv");


  //table1->Sort(0, sorted);



  return 0;
}


