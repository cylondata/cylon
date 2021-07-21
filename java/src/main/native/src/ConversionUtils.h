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

#ifndef CYLON_JNI_SRC_CONVERSIONUTILS_H_
#define CYLON_JNI_SRC_CONVERSIONUTILS_H_

#include <unordered_map>
#include <cylon/ctx/cylon_context.hpp>
#include <cylon/join/join_config.hpp>

extern std::unordered_map<int32_t, std::shared_ptr<cylon::CylonContext>> contexts;

extern std::unordered_map<std::string, cylon::join::config::JoinAlgorithm> join_algorithms;

extern std::unordered_map<std::string, cylon::join::config::JoinType> join_types;
#endif //CYLON_JNI_SRC_CONVERSIONUTILS_H_
