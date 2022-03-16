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

#include <cassert>
#include <iostream>

#include "gloo/mpi/context.h"
#include "gloo/transport/tcp/device.h"
#include "gloo/allreduce_ring.h"

int main(int argc, char** argv) {
  int rv;

  rv = MPI_Init(&argc, &argv);
  assert(rv == MPI_SUCCESS);

  // We'll use the TCP transport in this example
  auto dev = gloo::transport::tcp::CreateDevice("localhost");

  // Use inner scope to force destruction of context and algorithm
  {
    // Create Gloo context from MPI communicator
    auto context = std::make_shared<gloo::mpi::Context>(MPI_COMM_WORLD);
    context->connectFullMesh(dev);

    // Create and run simple allreduce
    int rank = context->rank;
    gloo::AllreduceRing<int> allreduce(context, {&rank}, 1);
    allreduce.run();
    std::cout << "Result: " << rank << std::endl;
  }

  rv = MPI_Finalize();
  assert(rv == MPI_SUCCESS);
  return 0;
}

