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

#ifndef CYLON_SRC_CYLON_CTX_CYLON_CONTEXT_HPP_
#define CYLON_SRC_CYLON_CTX_CYLON_CONTEXT_HPP_

#include <string>
#include "unordered_map"
#include "../net/comm_config.hpp"
#include "../net/communicator.hpp"
#include "memory_pool.hpp"

namespace cylon {
class CylonContext {
 private:
  std::unordered_map<std::string, std::string> config{};
  bool distributed;
  cylon::net::Communicator *communicator{};
  cylon::MemoryPool *memory_pool{};
  int32_t sequence_no = 0;

 public:
  static CylonContext *Init();
  void Finalize();

  static CylonContext *InitDistributed(net::CommConfig *config);
  void AddConfig(const std::string &key, const std::string &value);
  std::string GetConfig(const std::string &key, const std::string &def = "");
  net::Communicator *GetCommunicator() const;
  void setCommunicator(net::Communicator *communicator1);
  void setDistributed(bool distributed);
  int GetRank();
  int GetWorldSize();
  vector<int> GetNeighbours(bool include_self);
  explicit CylonContext(bool distributed);
  cylon::MemoryPool *GetMemoryPool();
  void SetMemoryPool(cylon::MemoryPool *mem_pool);
  int32_t GetNextSequence();
  void Barrier() {
    this->GetCommunicator()->Barrier();
  }
};
}

#endif //CYLON_SRC_CYLON_CTX_CYLON_CONTEXT_HPP_
