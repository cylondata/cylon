/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mpi.h>
#include <iostream>
#include "python/net/comm/all_to_all_wrap.h"
#include "python/net/distributed.h"

using namespace std;
using namespace cylon::net::comm;

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