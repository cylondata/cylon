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

#include "python/net/comm/callback.h"

bool cylon::net::comms::Callback::onReceive(int source, void *buffer, int length) {
  std::cout << "Received value: " << source << " length " << length << std::endl;
  delete[] reinterpret_cast<char *>(buffer);
  return false;
}

bool cylon::net::comms::Callback::onReceiveHeader(int source, int finished,
    int *buffer, int length) {
  std::cout << "Received HEADER: " << source << " length " << length << std::endl;
  return false;
}

bool cylon::net::comms::Callback::onSendComplete(int target, void *buffer, int length) {
  return false;
}
