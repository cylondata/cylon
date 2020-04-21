#include "builtins.h"
#include "iostream"

using namespace std;

void twisterx::util::printArray(void *buf, int size, string dataType, int depth) {
  if (dataType == "int") {
    if (depth == 8) {
      int8_t *int8Ptr = (int8_t *) buf;//getIntPointer<8>(buf);
      printInt8Array(int8Ptr, size);
    } else if (depth == 16) {
      int16_t *int16Ptr = (int16_t *) buf;//getIntPointer<16>(buf);
      printInt16Array(int16Ptr, size);
    } else if (depth == 32) {
      int32_t *int32Ptr = (int32_t *) buf;//getIntPointer<32>(buf);
      printInt32Array(int32Ptr, size);
    } else if (depth == 64) {
      int64_t *int64Ptr = (int64_t *) buf;//getIntPointer<64>(buf);
      printInt64Array(int64Ptr, size);
    }
  } else if (dataType == "float") {
    float *floatPtr = (float *) buf;//getFloatPointer(buf);
    printFloatArray(floatPtr, size);
  } else if (dataType == "double") {
    double *doublePtr = (double *) buf;//getFloatPointer(buf);
    printDoubleArray(doublePtr, size);
  } else if (dataType == "long") {
    long *longPtr = (long *) buf;//getFloatPointer(buf);
    printLongArray(longPtr, size);
  }

}
void twisterx::util::printInt8Array(int8_t *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}

void twisterx::util::printInt16Array(int16_t *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}
void twisterx::util::printInt32Array(int32_t *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}

void twisterx::util::printInt64Array(int64_t *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}

void twisterx::util::printFloatArray(float *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}

void twisterx::util::printDoubleArray(double *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}

void twisterx::util::printLongArray(long *buf, int size) {
  for (int i = 0; i < size; ++i) {
    std::cout << buf[i] << " ";
  }
  std::cout << std::endl;
}



