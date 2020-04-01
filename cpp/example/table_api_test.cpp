#include <io/Table.h>

int main(int argc, char *argv[]) {

  auto table = twisterx::io::Table::from_csv("/tmp/csv.csv");
  auto table2 = twisterx::io::Table::from_csv("/tmp/csv.csv");
  auto table3 = twisterx::io::Table::from_csv("/tmp/csv.csv");
  std::vector<std::shared_ptr<twisterx::io::Table>> tables;
  tables.push_back(table);
  tables.push_back(table);
  tables.push_back(table);
  auto tab = twisterx::io::Table::merge(tables);
  tab->print();
  return 0;
}
