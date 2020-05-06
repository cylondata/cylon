#include "util/builtins.h"
#include "net/TxRequest.h"

using namespace std;

int main() {

  int8_t int8buf[4] = {41, 42, 43, 44};
  int16_t int16buf[4] = {161, 162, 163, 164};
  int32_t int32buf[4] = {321, 322, 323, 324};
  int64_t int64buf[4] = {641, 642, 643, 644};
  float floatbuf[4] = {11.21, 12.10, 13.20, 14.20};
  int head[4] = {1, 2, 3, 4};

  //twisterx::util::printArray(int8buf, 4, "int", 8);
  twisterx::util::printArray(int16buf, 4, "int", 16);
  twisterx::util::printArray(int32buf, 4, "int", 32);
  twisterx::util::printArray(int64buf, 4, "int", 64);
  twisterx::util::printArray(floatbuf, 4, "float", -1);
  twisterx::TxRequest tx(10, int32buf, 4, head, 4);
  tx.to_string("int", 32);

  return 0;
}

