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

#include "iostream"
using namespace std;

namespace cylon {
class TxRequest {

 public:
  void *buffer{};
  int length{};
  int target;
  int header[6] = {};
  int headerLength{};

  TxRequest(int tgt, void *buf, int len);

  TxRequest(int tgt, void *buf, int len, int *head, int hLength);

  explicit TxRequest(int tgt);

  ~TxRequest();

  void to_string(string dataType, int bufDepth);
};
}  // namespace cylon

#endif //CYLON_TXREQUEST_H
