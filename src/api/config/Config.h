#ifndef TWISTERX_CONFIG_H
#define TWISTERX_CONFIG_H

#import <map>
#import <string>
#include <any>


namespace twister::config {
    class Config {

    private:
        std::map<std::string, std::any> config;

    public:
        std::string get_string(const std::string &key);

        void put_string(const std::string &key, const std::string &val);

        int get_int(const std::string &key);

        void put_int(const std::string &key, const int &val);
    };
}
#endif //TWISTERX_CONFIG_H
