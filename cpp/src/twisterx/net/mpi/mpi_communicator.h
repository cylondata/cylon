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

#ifndef TWISTERX_SRC_TWISTERX_COMM_MPICOMMUNICATOR_H_
#define TWISTERX_SRC_TWISTERX_COMM_MPICOMMUNICATOR_H_
#include "../comm_config.h"
#include "../communicator.h"
namespace cylon {
namespace net {

class MPIConfig : public CommConfig {
  // no configs for MPI. This is an example
  void DummyConfig(int dummy);

  int GetDummyConfig();

  CommType Type();

};

class MPICommunicator : public Communicator {
  void Init(CommConfig *config) override;
  Channel *CreateChannel() override;
  int GetRank() override;
  int GetWorldSize() override;
  void Finalize() override;
  void Barrier() override;
};
}
}
#endif //TWISTERX_SRC_TWISTERX_COMM_MPICOMMUNICATOR_H_
