#include <iostream>
#include "../include/org_twisterx_io_Table.h"
#include "io/table_api.h"

std::string jstr_to_str(JNIEnv *env, jstring jstr) {
  jboolean is_copy;
  return std::string(env->GetStringUTFChars(jstr, &is_copy));
}

void throwException(JNIEnv *env, const std::string &clazz, const std::string &msg) {
  std::cout << "throwing" << msg << std::endl;
  jclass exClass = env->FindClass(clazz.c_str());
  env->ThrowNew(exClass, msg.c_str());
}

void throwIOException(JNIEnv *env, const std::string &msg) {
  throwException(env, "java/io/IOException", msg);
}

JNIEXPORT void JNICALL Java_org_twisterx_io_Table_nativeJoin
    (JNIEnv *env, jclass thiz, jstring left_table, jstring right_table,
     jint left_join_col, jint right_join_col,
     jstring destination_table) {
  twisterx::io::join(
      jstr_to_str(env, left_table),
      jstr_to_str(env, right_table),
      left_join_col,
      right_join_col,
      jstr_to_str(env, destination_table)
  );
}

JNIEXPORT void JNICALL Java_org_twisterx_io_Table_nativeLoadCSV
    (JNIEnv *env, jclass thiz, jstring path, jstring uuid) {
  twisterx::io::Status status = twisterx::io::read_csv(jstr_to_str(env, path),
                                                       jstr_to_str(env, uuid));
  if (status.get_code() != twisterx::io::Code::OK) {
    throwIOException(env, status.get_msg());
  }
}

JNIEXPORT jint JNICALL Java_org_twisterx_io_Table_nativeColumnCount
    (JNIEnv *env, jclass thiz, jstring uuid) {
  return twisterx::io::column_count(jstr_to_str(env, uuid));
}

JNIEXPORT jint JNICALL Java_org_twisterx_io_Table_nativeRowCount
    (JNIEnv *env, jclass thiz, jstring uuid) {
  return twisterx::io::row_count(jstr_to_str(env, uuid));
}