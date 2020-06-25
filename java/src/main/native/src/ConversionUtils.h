#ifndef TWISTERX_JNI_SRC_CONVERSIONUTILS_H_
#define TWISTERX_JNI_SRC_CONVERSIONUTILS_H_

#include "unordered_map"
#include "ctx/twisterx_context.h"
#include "join/join_config.h"

extern std::unordered_map<int32_t, twisterx::TwisterXContext *> contexts;

extern std::unordered_map<std::string, twisterx::join::config::JoinAlgorithm> join_algorithms;

extern std::unordered_map<std::string, twisterx::join::config::JoinType> join_types;
#endif //TWISTERX_JNI_SRC_CONVERSIONUTILS_H_
