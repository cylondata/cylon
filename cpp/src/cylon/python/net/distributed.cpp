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
#include "python/net/distributed.h"

void cylon::net::cdist_init() {
  MPI_Init(NULL, NULL);
}

void cylon::net::cdist_finalize() {
  MPI_Finalize();
}

int cylon::net::cget_rank() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

int cylon::net::cget_size() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}


