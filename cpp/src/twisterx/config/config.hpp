#ifndef TWISTERX_CONFIG_H
#define TWISTERX_CONFIG_H

#include <unordered_map>
#include <string>
#include <any>

namespace twisterx::config {
  class Config {

  private:
    std::unordered_map<std::string, std::any> config;

  public:
    std::string get_string(const std::string &key);

    void put_string(const std::string &key, const std::string &val);

    int get_int(const std::string &key);

    void put_int(const std::string &key, const int &val);
  };
}
#endif //TWISTERX_CONFIG_H
