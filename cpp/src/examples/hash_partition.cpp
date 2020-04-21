#include <table.hpp>
#include <status.hpp>
#include <iostream>
#include <io/csv_read_config.h>

int main(int argc, char *argv[]) {
  std::shared_ptr<twisterx::Table> table1;
  auto status2 = twisterx::Table::FromCSV("/tmp/csv.csv", &table1);

  std::vector<std::shared_ptr<twisterx::Table>> outs;

  table1->print();

  table1->HashPartition({1}, 2, &outs);

  for (auto t: outs) {
    std::cout << "-----" << std::endl;
    t->print();
  }

  return 0;
}


