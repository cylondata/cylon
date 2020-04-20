#ifndef TWISTERX_SRC_UTIL_UUID_H_
#define TWISTERX_SRC_UTIL_UUID_H_

#include <random>
#include <sstream>

namespace twisterx {
namespace util {
namespace uuid {

/*
 * Generate a UUID
 */
std::string generate_uuid_v4();

}
}
}

#endif //TWISTERX_SRC_UTIL_UUID_H_
