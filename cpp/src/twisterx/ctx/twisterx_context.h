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

#ifndef TWISTERX_SRC_TWISTERX_CTX_TWISTERX_CONTEXT_H_
#define TWISTERX_SRC_TWISTERX_CTX_TWISTERX_CONTEXT_H_

#include <string>
#include "unordered_map"
#include "../net/comm_config.h"
#include "../net/communicator.h"
#include "memory_pool.h"

namespace twisterx {
class TwisterXContext {
 private:
  std::unordered_map<std::string, std::string> config{};
  bool distributed;
  twisterx::net::Communicator *communicator{};
  twisterx::MemoryPool *memory_pool;

 public:
  static TwisterXContext *Init();
  void Finalize();

  static TwisterXContext *InitDistributed(net::CommConfig *config);
  void AddConfig(const std::string &key, const std::string &value);
  std::string GetConfig(const std::string &key, const std::string &def = "");
  net::Communicator *GetCommunicator() const;
  void setCommunicator(net::Communicator *communicator1);
  void setDistributed(bool distributed);
  int GetRank();
  int GetWorldSize();
  vector<int> GetNeighbours(bool include_self);
  explicit TwisterXContext(bool distributed);

  template<typename TYPE>
  TYPE *GetMemoryPool();

  void SetMemoryPool(twisterx::MemoryPool* mem_pool);
};
}

#endif //TWISTERX_SRC_TWISTERX_CTX_TWISTERX_CONTEXT_H_
