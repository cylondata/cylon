#include "ArrowTable.h"
#include <org_cylon_arrow_ArrowTable.h>
#include <cstdint>
#include <iostream>

JNIEXPORT void JNICALL Java_org_cylon_arrow_ArrowTable_addColumn
    (JNIEnv *env, jclass clz, jstring table_id, jint data_type, jlong address, jlong size) {
  auto *buffer = reinterpret_cast<int8_t *>(address);
  for (int64_t i = 0; i < size;) {
    int a = int((unsigned char) (buffer[i + 3]) << 24 |
        (unsigned char) (buffer[i + 2]) << 16 |
        (unsigned char) (buffer[i + 1]) << 8 |
        (unsigned char) (buffer[i]));
    std::cout << a << ",";
    i += 4;
  }
  std::cout << std::endl;
}