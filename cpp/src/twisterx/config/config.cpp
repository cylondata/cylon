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

#include "config.hpp"

#include <map>

namespace twisterx::config {
std::string Config::get_string(const std::string &key) {
  return std::any_cast<std::string>(this->config[key]);
}

int Config::get_int(const std::string &key) {
  return std::any_cast<int>(this->config[key]);
}

void Config::put_string(const std::string &key, const std::string &val) {
  this->config.insert(std::pair<std::string, std::string>(key, val));
}

void Config::put_int(const std::string &key, const int &val) {
  this->config.insert(std::pair<std::string, int>(key, val));
}
}
