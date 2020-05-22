#include <table.hpp>
#include <status.hpp>
#include <iostream>
#include <io/csv_read_config.h>

using namespace twisterx;

int main(int argc, char *argv[]) {

  std::shared_ptr<twisterx::Table> table1, table2, joined;
  std::shared_ptr<arrow::Table> arwTable;
  std::shared_ptr<Table> *sorted;

  std::vector<std::string> columnnames;
  columnnames.push_back("c1");
  columnnames.push_back("c2");
  columnnames.push_back("c3");

  auto status1 = twisterx::Table::FromCSV("/tmp/csv.csv", &table1,
										  twisterx::io::config::CSVReadOptions()
										  .WithDelimiter(',')
										  .ColumnNames(columnnames));
  auto status2 = twisterx::Table::FromCSV("/tmp/csv.csv", &table2,
										  twisterx::io::config::CSVReadOptions()
											  .WithDelimiter(',')
											  .ColumnNames(columnnames));

  table1->ToArrowTable(arwTable);


  std::cout << "Table1 Columns : " << arwTable->num_columns() << std::endl;

  for (int kI = 0; kI < arwTable->schema()->num_fields(); ++kI) {
	std::cout << "Filed : " << kI << ": " << arwTable->schema()->field(kI)->ToString()<< std::endl;
  }

  table1->Join(table2,
			   twisterx::join::config::JoinConfig::RightJoin(0, 1),
			   &joined);
  joined->print();
  joined->WriteCSV("/tmp/out.csv");


  //table1->Sort(0, sorted);



  return 0;
}


