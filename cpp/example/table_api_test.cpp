#include <iostream>
#include "io/table_api.h"

int main(int argc, char *argv[]) {
  twisterx::io::read_csv("/tmp/csv.csv", "1");

  twisterx::io::print("1", 0, twisterx::io::column_count("1"), 0, twisterx::io::row_count("1"));
  return 0;
}
