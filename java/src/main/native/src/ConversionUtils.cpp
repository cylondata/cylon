/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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