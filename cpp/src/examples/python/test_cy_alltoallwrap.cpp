#include <mpi.h>
#include <iostream>
#include "python/net/comm/all_to_all_wrap.h"
#include "python/net/distributed.h"

using namespace std;
using namespace twisterx::net::comm;

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
//
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::vector<int> sources;
  std::vector<int> targets;
  for (int i = 0; i < size; i++) {
	sources.push_back(i);
	targets.push_back(i);
  }
  int *buf = new int[4]{4};
  int *header = new int[4];
  header[0] = 1;
  header[1] = 2;
  header[2] = 3;
  header[3] = 4;

  all_to_all_wrap *all_wrap = new all_to_all_wrap(rank, sources, targets, 1);
  all_wrap->insert(buf, 4, 0, header, 4);
  all_wrap->wait();

  delete[] header;
  delete[] buf;
  all_wrap->finish();
  MPI_Finalize();
  return 0;

}