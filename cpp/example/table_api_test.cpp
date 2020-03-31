#include <io/Table.h>

int main(int argc, char *argv[]) {

  auto table = twisterx::io::Table::from_csv("/tmp/csv.csv");
  table.print();
  return 0;
}
