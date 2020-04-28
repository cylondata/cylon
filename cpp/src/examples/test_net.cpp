//
// Created by vibhatha on 4/26/20.
//

#include "python/net/distributed.h"

using namespace std;

int main(int argc, char *argv[]) {
  twisterx::net::cdist_init();
  int rank = twisterx::net::cget_rank();
  int size = twisterx::net::cget_size();
  cout << "Rank : " << rank << ", Size : " << size << endl;
  twisterx::net::cdist_finalize();
}