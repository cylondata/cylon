#include "io/table_api.h"

int main(int argc, char *argv[]) {
  twisterx::io::read_csv("/tmp/csv.csv", "123");
  return 0;
}
