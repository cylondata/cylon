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

#ifndef CYLON_SRC_UTIL_UUID_H_
#define CYLON_SRC_UTIL_UUID_H_

#include <random>
#include <sstream>

namespace cylon {
namespace util {

/*
 * Generate a UUID
 */
std::string generate_uuid_v4();

}  // namespace util
}  // namespace cylon

#endif //CYLON_SRC_UTIL_UUID_H_
