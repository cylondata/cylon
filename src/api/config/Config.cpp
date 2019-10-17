#include "Config.h"

#include <map>

namespace twister::config {

    string Config::get_string(const string &key) {
        return any_cast<string>(this->config[key]);
    }

    int Config::get_int(const string &key) {
        return any_cast<int>(this->config[key]);
    }

    void Config::put_string(const string &key, const string &val) {
        this->config.insert(pair<string, string>(key, val));
    }

    void Config::put_int(const string &key, const int &val) {
        this->config.insert(pair<string, int>(key, val));
    }
}