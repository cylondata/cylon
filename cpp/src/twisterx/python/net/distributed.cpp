#include <mpi.h>
#include "distributed.h"

void twisterx::net::cdist_init() {
  MPI_Init(NULL, NULL);
}

void twisterx::net::cdist_finalize() {
  MPI_Finalize();
}

int twisterx::net::cget_rank() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int twisterx::net::cget_size() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}

