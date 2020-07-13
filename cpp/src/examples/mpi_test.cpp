#include <mpi.h>
#include <cstdio>
#include <iostream>

int main(int argc, char *argv[]) {
  int x = 0;
  std::cout << (int64_t)(&x - 1) << std::endl;
  std::cout << (int64_t)(&x) << std::endl;
}