#include <mpi.h>
#include <iostream>

#include "net/all_to_all.hpp"


class Clbk : public twisterx::ReceiveCallback {
 public:
  bool onReceive(int source, void *buffer, int length) override {
    std::cout << "Received value: " << source << " length " << length << std::endl;
    delete[] reinterpret_cast<char *>(buffer);
    return false;
  }

  bool onReceiveHeader(int source, int finished, int *buffer, int length) override {
    std::cout << "Received HEADER: " << source << " length " << length << std::endl;
    return false;
  }

  bool onSendComplete(int target, void *buffer, int length) override {
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
  std::cout << "Starting the all receive - " << rank << std::endl;
  int buf[4] = {rank};
  int *header = new int[4];
  header[0] = 1;
  header[1] = 2;
  header[2] = 3;
  header[3] = 4;
  std::cout << "Size : " << size  << ", Rank : " << rank << std::endl;
  for (int i = 0; i < size * 1; i++) {
    all.insert(buf, 4, i % size, header, 4);
  }
  all.finish();
//
  while (!all.isComplete()) {
  }

  delete[] header;
  all.close();
  MPI_Finalize();
  return 0;
}
