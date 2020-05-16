//
// Created by vibhatha on 5/15/20.
//

#ifndef TWISTERX_SRC_TWISTERX_PYTHON_CTX_TWISTERX_CONTEXT_WRAP_H_
#define TWISTERX_SRC_TWISTERX_PYTHON_CTX_TWISTERX_CONTEXT_WRAP_H_

#include <string>
#include "unordered_map"
#include "../../net/comm_config.h"
#include "../../net/communicator.h"
#include "../../ctx/twisterx_context.h"

using namespace twisterx;

namespace twisterx {
namespace py {
class twisterx_context_wrap {
 private:
  std::unordered_map<std::string, std::string> config{};

  bool distributed;

  twisterx::net::Communicator *communicator{};

  explicit twisterx_context_wrap(bool distributed);

 public:
  static TwisterXContext *Init();
  void Finalize();

  static TwisterXContext *InitDistributed(std::string config);
  void AddConfig(const std::string &key, const std::string &value);
  std::string GetConfig(const std::string &key, const std::string &def = "");
  net::Communicator *GetCommunicator() const;
  int GetRank();
  int GetWorldSize();
  vector<int> GetNeighbours(bool include_self);

};
}
}

#endif //TWISTERX_SRC_TWISTERX_PYTHON_CTX_TWISTERX_CONTEXT_WRAP_H_
