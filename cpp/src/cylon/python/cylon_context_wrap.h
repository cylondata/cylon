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

#ifndef CYLON_SRC_CYLON_PYTHON_CTX_CYLON_CONTEXT_WRAP_H_
#define CYLON_SRC_CYLON_PYTHON_CTX_CYLON_CONTEXT_WRAP_H_

#include <string>
#include "unordered_map"
#include "../net/comm_config.hpp"
#include "../net/communicator.hpp"
#include "../ctx/cylon_context.hpp"

namespace cylon {
namespace python {
class cylon_context_wrap {
 private:
  std::unordered_map<std::string, std::string> config{};
     
  bool distributed;

  cylon::net::Communicator *communicator{};

  CylonContext *context;

  explicit cylon_context_wrap(bool distributed);

  cylon::MemoryPool *memory_pool{};

  int32_t sequence_no;


 public:

  cylon_context_wrap();

  cylon_context_wrap(std::string config);

  CylonContext *getInstance();

//  static CylonContext *Init();
//
//  static CylonContext *InitDistributed(std::string config);

  void AddConfig(const std::string &key, const std::string &value);

  std::string GetConfig(const std::string &key, const std::string &defn = "");

  net::Communicator *GetCommunicator() const;

  void Barrier();

  int GetRank();

  int GetWorldSize();

  void Finalize();

  int GetContextId();

  vector<int> GetNeighbours(bool include_self);

  cylon::MemoryPool *GetMemoryPool();

  void SetMemoryPool(cylon::MemoryPool *mem_pool);
  
  int32_t GetNextSequence();

};
}
}

#endif //CYLON_SRC_CYLON_PYTHON_CTX_CYLON_CONTEXT_WRAP_H_
