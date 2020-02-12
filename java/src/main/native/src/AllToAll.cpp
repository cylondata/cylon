#include "AllToAll.h"

#include "../include/org_twister2_AllToAll.h"
#include <iostream>

JNIEXPORT void JNICALL Java_org_twister2_AllToAll_WriteMessage(JNIEnv *, jobject) {
  std::cout << "Hello JNI";
}