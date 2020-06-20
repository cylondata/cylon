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

#ifndef TWISTERX_SRC_TWISTERX_PYTHON_CTX_TWISTERX_CONTEXT_WRAP_H_
#define TWISTERX_SRC_TWISTERX_PYTHON_CTX_TWISTERX_CONTEXT_WRAP_H_

#include <string>
#include "unordered_map"
#include "../net/comm_config.h"
#include "../net/communicator.h"
#include "../ctx/twisterx_context.h"

using namespace twisterx;

namespace twisterx {
namespace python {
class twisterx_context_wrap {
 private:
  std::unordered_map<std::string, std::string> config{};
     
  bool distributed;

  twisterx::net::Communicator *communicator{};

  TwisterXContext *context;

  explicit twisterx_context_wrap(bool distributed);

  twisterx::MemoryPool *memory_pool{};

  int32_t sequence_no;


 public:

  twisterx_context_wrap();

  twisterx_context_wrap(std::string config);

  TwisterXContext *getInstance();

//  static TwisterXContext *Init();
//
//  static TwisterXContext *InitDistributed(std::string config);

  void AddConfig(const std::string &key, const std::string &value);

  std::string GetConfig(const std::string &key, const std::string &defn = "");

  net::Communicator *GetCommunicator() const;

  int GetRank();

  int GetWorldSize();

  void Finalize();

  int GetContextId();

  vector<int> GetNeighbours(bool include_self);

  twisterx::MemoryPool *GetMemoryPool();

  void SetMemoryPool(twisterx::MemoryPool *mem_pool);
  
  int32_t GetNextSequence();

};
}
}

#endif //TWISTERX_SRC_TWISTERX_PYTHON_CTX_TWISTERX_CONTEXT_WRAP_H_
