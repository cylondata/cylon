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

  TwisterXContext *context;

  explicit twisterx_context_wrap(bool distributed);

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

  vector<int> GetNeighbours(bool include_self);

};
}
}

#endif //TWISTERX_SRC_TWISTERX_PYTHON_CTX_TWISTERX_CONTEXT_WRAP_H_
