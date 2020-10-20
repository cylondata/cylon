#ifndef CYLON_JNI_SRC_CONVERSIONUTILS_H_
#define CYLON_JNI_SRC_CONVERSIONUTILS_H_

#include "unordered_map"
#include "ctx/cylon_context.hpp"
#include "join/join_config.hpp"

extern std::unordered_map<int32_t, std::shared_ptr<cylon::CylonContext>> contexts;

extern std::unordered_map<std::string, cylon::join::config::JoinAlgorithm> join_algorithms;

extern std::unordered_map<std::string, cylon::join::config::JoinType> join_types;
#endif //CYLON_JNI_SRC_CONVERSIONUTILS_H_
