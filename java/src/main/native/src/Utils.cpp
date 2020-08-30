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

#include "Utils.hpp"

std::string jstr_to_str(JNIEnv *env, jstring jstr) {
  jboolean is_copy;
  return std::string(env->GetStringUTFChars(jstr, &is_copy));
}

void throwException(JNIEnv *env, const std::string &clazz, const std::string &msg) {
  jclass exClass = env->FindClass(clazz.c_str());
  env->ThrowNew(exClass, msg.c_str());
}

void throwCylonRuntimeException(JNIEnv *env, const std::string &msg) {
  throwException(env, "org/cylondata/cylon/exception/CylonRuntimeException", msg);
}
