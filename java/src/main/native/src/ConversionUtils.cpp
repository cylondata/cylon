#include <cstdint>
#include <ctx/twisterx_context.h>
#include <join/join_config.h>
#include "ConversionUtils.h"

std::unordered_map<int32_t, twisterx::TwisterXContext *> contexts{};

std::unordered_map<std::string, twisterx::join::config::JoinAlgorithm> join_algorithms{
    std::pair<std::string, twisterx::join::config::JoinAlgorithm>("HASH", twisterx::join::config::JoinAlgorithm::HASH),
    std::pair<std::string, twisterx::join::config::JoinAlgorithm>("SORT", twisterx::join::config::JoinAlgorithm::SORT)
};

//LEFT, RIGHT, INNER, FULL_OUTER
std::unordered_map<std::string, twisterx::join::config::JoinType> join_types{
    std::pair<std::string, twisterx::join::config::JoinType>("LEFT", twisterx::join::config::JoinType::LEFT),
    std::pair<std::string, twisterx::join::config::JoinType>("RIGHT", twisterx::join::config::JoinType::RIGHT),
    std::pair<std::string, twisterx::join::config::JoinType>("INNER", twisterx::join::config::JoinType::INNER),
    std::pair<std::string, twisterx::join::config::JoinType>("FULL_OUTER", twisterx::join::config::JoinType::FULL_OUTER)
};