#ifndef TWISTERX_SRC_TWISTERX_COMM_COMM_CONFIG_H_
#define TWISTERX_SRC_TWISTERX_COMM_COMM_CONFIG_H_
#include <string>
#include <unordered_map>
#include "comm_type.h"
namespace twisterx {
namespace net {
class CommConfig {
 private:
  std::unordered_map<std::string, void *> config;

 protected:
  void AddConfig(const std::string &key, void *value) {
	this->config.insert(std::pair<std::string, void *>(key, value));
  }

  void *GetConfig(const std::string &key) {
	return this->config.find(key)->second;
  }
 public:
  virtual CommType Type() = 0;
};
}
}

#endif //TWISTERX_SRC_TWISTERX_COMM_COMM_CONFIG_H_
