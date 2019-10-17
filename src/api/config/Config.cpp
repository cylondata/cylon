#include "Config.h"

#include <map>

namespace twister::config {

    std::string Config::get_string(const std::string &key) {
        return std::any_cast<std::string>(this->config[key]);
    }

    int Config::get_int(const std::string &key) {
        return std::any_cast<int>(this->config[key]);
    }

    void Config::put_string(const std::string &key, const std::string &val) {
        this->config.insert(std::pair<std::string, std::string>(key, val));
    }

    void Config::put_int(const std::string &key, const int &val) {
        this->config.insert(std::pair<std::string, int>(key, val));
    }
}