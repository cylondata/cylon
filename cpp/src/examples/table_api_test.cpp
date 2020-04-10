#include <table.hpp>
#include <status.hpp>
#include <iostream>

void change(std::shared_ptr<std::string> &str) {
  str.reset(new std::string("bye"));
}

int main(int argc, char *argv[]) {
//  std::unique_ptr<twisterx::Table> table1, table2, table3, table4;
//  auto status = twisterx::Table::FromCSV("/tmp/csv.csv", &table1);
//  status = twisterx::Table::FromCSV("/tmp/csv.csv", &table2);
//  status = twisterx::Table::FromCSV("/tmp/csv.csv", &table3);
//  std::vector<std::shared_ptr<twisterx::Table>> tables;
//  tables.push_back(std::move(table1));
//  tables.push_back(std::move(table2));
//  tables.push_back(std::move(table3));
//  twisterx::Table::Merge(tables, &table4);
//  table4->WriteCSV("/tmp/out.csv");

  auto i1 = std::make_shared<std::string>("hi");
  const auto &i2 = i1;

  change(i1);

  std::cout << i1->c_str() << "," << i2->c_str();

  return 0;
}


