#ifndef TWISTERX_MURMUR3_H
#define TWISTERX_MURMUR3_H

#include <memory>
#include <string>

namespace twisterx {
namespace util {

void MurmurHash3_x86_32(const void *key, int len, uint32_t seed, void *out);

void MurmurHash3_x86_128(const void *key, int len, uint32_t seed, void *out);

void MurmurHash3_x64_128(const void *key, int len, uint32_t seed, void *out);
}
}

#endif //TWISTERX_MURMER3_H
