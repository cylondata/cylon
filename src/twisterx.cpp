#include <iostream>
#include <mpi.h>

int main() {
  MPI_Init(NULL, NULL);
  std::cout << "Starting TwisterX program ..." << std::endl;
  
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Finalize();
}
