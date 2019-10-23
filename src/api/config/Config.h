#ifndef TWISTERX_CONFIG_H
#define TWISTERX_CONFIG_H

#include <unordered_map>
#include <string>
#include <any>

using namespace std;

namespace twister::config {
    class Config {

    private:
        std::unordered_map<string, any> config;

    public:
        string get_string(const string &key);

        void put_string(const string &key, const string &val);

        int get_int(const string &key);

        void put_int(const string &key, const int &val);
    };
}

#endif //TWISTERX_CONFIG_H
