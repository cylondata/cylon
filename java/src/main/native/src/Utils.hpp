#ifndef CYLON_JNI_SRC_UTILS_HPP_
#define CYLON_JNI_SRC_UTILS_HPP_
#include <jni.h>
#include <string>
std::string jstr_to_str(JNIEnv *env, jstring jstr) {
  jboolean is_copy;
  return std::string(env->GetStringUTFChars(jstr, &is_copy));
}

void throwException(JNIEnv *env, const std::string &clazz, const std::string &msg) {
  jclass exClass = env->FindClass(clazz.c_str());
  env->ThrowNew(exClass, msg.c_str());
}
#endif //CYLON_JNI_SRC_UTILS_HPP_
