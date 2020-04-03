#ifndef TWISTERX_SRC_UTIL_UUID_H_
#define TWISTERX_SRC_UTIL_UUID_H_

#include <random>
#include <sstream>

namespace twisterx {
    namespace util {
        namespace uuid {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_int_distribution<> dis(0, 15);
            static std::uniform_int_distribution<> dis2(8, 11);
            std::string generate_uuid_v4();

        }
    }
}

#endif //TWISTERX_SRC_UTIL_UUID_H_
