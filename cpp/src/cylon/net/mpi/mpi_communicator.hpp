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

#ifndef CYLON_SRC_CYLON_COMM_MPICOMMUNICATOR_H_
#define CYLON_SRC_CYLON_COMM_MPICOMMUNICATOR_H_

#include <cylon/net/comm_config.hpp>
#include <cylon/net/communicator.hpp>

namespace cylon {
namespace net {

class MPIConfig : public CommConfig {
 private:
  // no configs for MPI. This is an example
  void DummyConfig(int dummy);
  int GetDummyConfig();
public:
  CommType Type() override;
  ~MPIConfig() override;
  static std::shared_ptr<MPIConfig> Make();
};

class MPICommunicator : public Communicator {
 public:
  ~MPICommunicator() override = default;
  Status Init(const std::shared_ptr<CommConfig> &config) override;
  std::unique_ptr<Channel> CreateChannel() const override;
  int GetRank() const override;
  int GetWorldSize() const override;
  void Finalize() override;
  void Barrier() override;
  CommType GetCommType() const override;
};

class MPISyncCommunicator : public SyncCommunicator {
 public:
  Status AllGather(const std::shared_ptr<Table> &table,
                   std::vector<std::shared_ptr<Table>> *out) const override;

  Status Gather(const std::shared_ptr<Table> &table, int gather_root,
                bool gather_from_root, std::vector<std::shared_ptr<Table>> *out) const override;

  Status Bcast(const std::shared_ptr<Table> &table, int bcast_root,
               std::shared_ptr<Table> &out) const override;
};

}
}
#endif //CYLON_SRC_CYLON_COMM_MPICOMMUNICATOR_H_
