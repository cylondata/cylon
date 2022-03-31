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

#ifndef CYLON_TXREQUEST_H
#define CYLON_TXREQUEST_H

#include <iostream>

namespace cylon {
class CylonRequest {

 public:
  const void *buffer{};
  int length{};
  int target;
  int header[6] = {};
  int headerLength{};

  CylonRequest(int tgt, const void *buf, int len);

  CylonRequest(int tgt, const void *buf, int len, int *head, int hLength);

  explicit CylonRequest(int tgt);

  ~CylonRequest();

  void to_string(std::string dataType, int bufDepth);
};
}  // namespace cylon

#endif //CYLON_TXREQUEST_H
