#include <mpi.h>
#include <iostream>

#include "small_twister.hpp"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // ideally should use a buffer pool ...
  unsigned char* buffer = new unsigned char[1000];
  uint32_t start = 0;
  record_t<uint32_t, uint64_t, const char*> r1(buffer, &start, 5, 6, "sdasd");
  record_t<uint32_t, uint64_t, float, const char*> r2(buffer, &start, 7, 9, 1.03, "sddfsasd");

  std::cout << "[INFO] Starting the program ..." << std::endl;

  MPI_Finalize();
}
