#include <iostream>
#include <vector>
#include "../include/org_twisterx_Table.h"
#include "table_api.h"

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

JNIEXPORT void JNICALL Java_org_twisterx_Table_nativeJoin
    (JNIEnv *env, jclass thiz, jstring left_table, jstring right_table,
     jint left_join_col, jint right_join_col,
     jstring destination_table) {
  twisterx::joinTables(
      jstr_to_str(env, left_table),
      jstr_to_str(env, right_table),
      left_join_col,
      right_join_col,
      jstr_to_str(env, destination_table)
  );
}

JNIEXPORT void JNICALL Java_org_twisterx_Table_nativeLoadCSV
    (JNIEnv *env, jclass thiz, jstring path, jstring uuid) {
  twisterx::Status status = twisterx::read_csv(jstr_to_str(env, path),
                                                       jstr_to_str(env, uuid));
  if (!status.is_ok()) {
    throwIOException(env, status.get_msg());
  }
}

JNIEXPORT jint JNICALL Java_org_twisterx_Table_nativeColumnCount
    (JNIEnv *env, jclass thiz, jstring uuid) {
  return twisterx::column_count(jstr_to_str(env, uuid));
}

JNIEXPORT jint JNICALL Java_org_twisterx_Table_nativeRowCount
    (JNIEnv *env, jclass thiz, jstring uuid) {
  return twisterx::row_count(jstr_to_str(env, uuid));
}

JNIEXPORT void JNICALL Java_org_twisterx_Table_print
    (JNIEnv *env, jclass thiz, jstring uuid, jint row1, jint row2, jint col1, jint col2) {
  twisterx::print(jstr_to_str(env, uuid), col1, col2, row1, row2);
}

JNIEXPORT void JNICALL Java_org_twisterx_Table_merge
    (JNIEnv *env, jclass thiz, jobjectArray table_ids, jstring merge_tab_id) {
  int table_count = env->GetArrayLength(table_ids);

  std::vector<std::string> table_ids_vector;
  for (int i = 0; i < table_count; i++) {
    auto tab_id = (jstring) (env->GetObjectArrayElement(table_ids, i));
    table_ids_vector.push_back(jstr_to_str(env, tab_id));
  }
  twisterx::Status status = twisterx::merge(table_ids_vector, jstr_to_str(env, merge_tab_id));
  std::cout << status.get_code() << std::endl;
  if (!status.is_ok()) {
    throwIOException(env, status.get_msg());
  }
}