#include <cstdint>
#include <ctx/cylon_context.hpp>
#include <join/join_config.hpp>
#include "ConversionUtils.h"

std::unordered_map<int32_t, cylon::CylonContext *> contexts{};

std::unordered_map<std::string, cylon::join::config::JoinAlgorithm> join_algorithms{
    std::pair<std::string, cylon::join::config::JoinAlgorithm>("HASH", cylon::join::config::JoinAlgorithm::HASH),
    std::pair<std::string, cylon::join::config::JoinAlgorithm>("SORT", cylon::join::config::JoinAlgorithm::SORT)
};

//LEFT, RIGHT, INNER, FULL_OUTER
std::unordered_map<std::string, cylon::join::config::JoinType> join_types{
    std::pair<std::string, cylon::join::config::JoinType>("LEFT", cylon::join::config::JoinType::LEFT),
    std::pair<std::string, cylon::join::config::JoinType>("RIGHT", cylon::join::config::JoinType::RIGHT),
    std::pair<std::string, cylon::join::config::JoinType>("INNER", cylon::join::config::JoinType::INNER),
    std::pair<std::string, cylon::join::config::JoinType>("FULL_OUTER", cylon::join::config::JoinType::FULL_OUTER)
};