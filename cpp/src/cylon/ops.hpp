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

#ifndef CYLON_API_H
#define CYLON_API_H

#include "ops/dis_join_op.hpp"
#include "ops/dis_set_op.hpp"

namespace cylon {

/**
 * Union operation using the ops framework
 * @param ctx the context with the information about the environment
 * @param first left table
 * @param second right table
 * @param join_config join config with specification of the join
 * @param out the resulting table
 * @return the Cylon Status the status of the join operation
 */
Status JoinOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                         std::shared_ptr <cylon::Table> &first,
                         std::shared_ptr <cylon::Table> &second,
                         const cylon::join::config::JoinConfig &join_config,
                         std::shared_ptr <cylon::Table> &out);

/**
 * Union operation using the ops framework, left + right is the semantics with duplicates removed
 * @param ctx the context with the information about the environment
 * @param first first table
 * @param second second table
 * @param join_config join config with specification of the join
 * @param out the resulting table
 * @return the Cylon Status the status of the join operation
 */
Status UnionOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                          std::shared_ptr <cylon::Table> &first,
                          std::shared_ptr <cylon::Table> &second,
                          std::shared_ptr <cylon::Table> &out);

/**
 * Subtract operation using the ops framework, left - right is the semantics
 * @param ctx the context with the information about the environment
 * @param first first table
 * @param second second table
 * @param out the resulting table
 * @return the Cylon Status the status of the join operation
 */
Status SubtractOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                             std::shared_ptr <cylon::Table> &first,
                             std::shared_ptr <cylon::Table> &second,
                             std::shared_ptr <cylon::Table> &out);

/**
 * Intersect operation using the ops framework, left intersect right is the semantics
 * @param ctx the context with the information about the environment
 * @param first first table
 * @param second second table
 * @param out the resulting table
 * @return the Cylon Status the status of the join operation
 */
Status IntersectOperation(const std::shared_ptr <cylon::CylonContext> &ctx,
                              std::shared_ptr <cylon::Table> &first,
                              std::shared_ptr <cylon::Table> &second,
                              std::shared_ptr <cylon::Table> &out);
}

#endif //CYLON_API_H
