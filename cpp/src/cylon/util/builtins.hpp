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

#ifndef CYLON_BUILTINS_H
#define CYLON_BUILTINS_H

#include "iostream"

using namespace std;

namespace cylon {
namespace util {

void printArray(void *buf, int size, string dataType, int depth);

void printInt8Array(int8_t *buf, int size);

void printInt16Array(int16_t *buf, int size);

void printInt32Array(int32_t *buf, int size);

void printInt64Array(int64_t *buf, int size);

void printFloatArray(float *buf, int size);

void printDoubleArray(double *buf, int size);

void printLongArray(int64_t *buf, int size);
}  // namespace util
}  // namespace cylon

#endif //CYLON_BUILTINS_H
