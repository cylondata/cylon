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

#include "builtins.hpp"
#include <string>
#include <iostream>

void cylon::util::printArray(const void *buf, int size, std::string dataType, int depth) {
  if (dataType == "int") {
    if (depth == 8) {
      const int8_t *int8Ptr = reinterpret_cast<const int8_t *>(buf);  // getIntPointer<8>(buf);
      printInt8Array(int8Ptr, size);
    } else if (depth == 16) {
      const int16_t *int16Ptr = reinterpret_cast<const int16_t *>(buf);  // getIntPointer<16>(buf);
      printInt16Array(int16Ptr, size);
    } else if (depth == 32) {
      const int32_t *int32Ptr = reinterpret_cast<const int32_t *>(buf);  // getIntPointer<32>(buf);
      printInt32Array(int32Ptr, size);
    } else if (depth == 64) {
      const int64_t *int64Ptr = reinterpret_cast<const int64_t *>(buf);  // getIntPointer<64>(buf);
      printInt64Array(int64Ptr, size);
    }
  } else if (dataType == "float") {
    const float *floatPtr = reinterpret_cast<const float *>(buf);  // getFloatPointer(buf);
    printFloatArray(floatPtr, size);
  } else if (dataType == "double") {
    const double *doublePtr = reinterpret_cast<const double *>(buf);  // getFloatPointer(buf);
    printDoubleArray(doublePtr, size);
  } else if (dataType == "long") {
    const int64_t *longPtr = reinterpret_cast<const int64_t *>(buf);  // getFloatPointer(buf);
    printLongArray(longPtr, size);
  }
}

void cylon::util::printInt8Array(const int8_t *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}

void cylon::util::printInt16Array(const int16_t *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}

void cylon::util::printInt32Array(const int32_t *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}

void cylon::util::printInt64Array(const int64_t *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}

void cylon::util::printFloatArray(const float *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}

void cylon::util::printDoubleArray(const double *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}

void cylon::util::printLongArray(const int64_t *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}



