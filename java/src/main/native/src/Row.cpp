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

#include <row.hpp>
#include "org_cylondata_cylon_Row.h"

cylon::Row *get_cpp_row_address(JNIEnv *env, jobject *obj) {
  jclass row_cls = env->FindClass("org/cylondata/cylon/ops/Row");
  jfieldID row_id_field = env->GetFieldID(row_cls, "memoryAddress", "J");
  int64_t address = env->GetLongField(*obj, row_id_field);
  return reinterpret_cast<cylon::Row *>(address);
}

JNIEXPORT jbyte JNICALL Java_org_cylondata_cylon_ops_Row_getInt8
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return row->GetInt8(column_id);
}

JNIEXPORT jshort JNICALL Java_org_cylondata_cylon_ops_Row_getUInt8
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return row->GetUInt8(column_id);
}

JNIEXPORT jshort JNICALL Java_org_cylondata_cylon_ops_Row_getInt16
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return row->GetInt16(column_id);
}

JNIEXPORT jint JNICALL Java_org_cylondata_cylon_ops_Row_getUInt16
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return row->GetInt16(column_id);
}

JNIEXPORT jint JNICALL Java_org_cylondata_cylon_ops_Row_getInt32
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return row->GetInt32(column_id);
}

JNIEXPORT jlong JNICALL Java_org_cylondata_cylon_ops_Row_getUInt32
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return row->GetUInt32(column_id);
}

JNIEXPORT jlong JNICALL Java_org_cylondata_cylon_ops_Row_getInt64
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return row->GetInt64(column_id);
}

JNIEXPORT jlong JNICALL Java_org_cylondata_cylon_ops_Row_getUInt64
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return row->GetInt64(column_id);
}

JNIEXPORT jfloat JNICALL Java_org_cylondata_cylon_ops_Row_getHalfFloat
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return row->GetHalfFloat(column_id);
}

JNIEXPORT jfloat JNICALL Java_org_cylondata_cylon_ops_Row_getFloat
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return row->GetFloat(column_id);
}

JNIEXPORT jdouble JNICALL Java_org_cylondata_cylon_ops_Row_getDouble
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return row->GetDouble(column_id);
}

JNIEXPORT jstring JNICALL Java_org_cylondata_cylon_ops_Row_getString
    (JNIEnv *env, jobject obj, jlong column_id) {
  cylon::Row *row = get_cpp_row_address(env, &obj);
  return env->NewStringUTF(row->GetString(column_id).c_str());
}