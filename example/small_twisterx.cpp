#include <mpi.h>
#include <iostream>

#include "../include/all_to_all.hpp"

class Clbk : public twisterx::ReceiveCallback {
public:
  bool onReceive(int source, void *buffer, int length) override {
    std::cout << "Received value: " << source << " length " << length;
    return false;
  }
};

int main(int argc, char *argv[]) {
  std::cout << "First - ";
  MPI_Init(NULL, NULL);
//
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::cout << "First - " << rank << " size " << size << std::endl;
  std::vector<int> sources;
  std::vector<int> targets;
  for (int i = 0; i < size; i++) {
    sources.push_back(i);
    targets.push_back(i);
  }
  Clbk c;
  // ideally should use a buffer pool ...
  twisterx::AllToAll all(rank, sources, targets, 1, &c);
  std::cout << "Starting the all receive - " << rank;
  int buf[4] = {rank};
  for (int i = 0; i < size; i++) {
    all.insert(buf, 16, i);
  }
  all.finish();
//
  while (!all.isComplete()) {
  }
//
  MPI_Finalize();
  return 0;
}
