#include <iostream>
#include "io/table_api.h"

int main(int argc, char *argv[]) {
  twisterx::io::read_csv("/tmp/csv.csv", "123");
  int columns = twisterx::io::column_count("123");
  int rows = twisterx::io::row_count("123");
  std::cout << rows << "," << columns << std::endl;
  return 0;
}
