#include <table.hpp>

int main(int argc, char *argv[]) {

  auto table = twisterx::Table::from_csv("/tmp/csv.csv");
  auto table2 = twisterx::Table::from_csv("/tmp/csv.csv");
  auto table3 = twisterx::Table::from_csv("/tmp/csv.csv");
  std::vector<std::shared_ptr<twisterx::Table>> tables;
  tables.push_back(table);
  tables.push_back(table);
  tables.push_back(table);
  auto tab = twisterx::Table::merge(tables);
  tab->print();
  return 0;
}
