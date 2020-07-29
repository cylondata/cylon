#ifndef CYLON_JNI_SRC_UTILS_HPP_
#define CYLON_JNI_SRC_UTILS_HPP_
#include <jni.h>
#include <string>

std::string jstr_to_str(JNIEnv *env, jstring jstr);
void throwException(JNIEnv *env, const std::string &clazz, const std::string &msg);
void throwCylonRuntimeException(JNIEnv *env, const std::string &msg);
#endif //CYLON_JNI_SRC_UTILS_HPP_
