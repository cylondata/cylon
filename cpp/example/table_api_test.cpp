#include <iostream>
#include "io/table_api.h"

int main(int argc, char *argv[]) {
  twisterx::io::read_csv("/tmp/csv.csv", "1");
  twisterx::io::read_csv("/tmp/csv.csv", "2");

  twisterx::io::join("1", "2", 0, 0, "3");

  int columns = twisterx::io::column_count("3");
  int rows = twisterx::io::row_count("3");
  std::cout << rows << "," << columns << std::endl;
  return 0;
}
