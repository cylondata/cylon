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

#include <iostream>
#include <vector>
#include "org_cylondata_cylon_Table.h"
#include "table_api.hpp"
#include "ConversionUtils.h"
#include "Utils.hpp"

void throwIOException(JNIEnv *env, const std::string &msg) {
  throwException(env, "java/io/IOException", msg);
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_Table_nativeJoin
    (JNIEnv *env, jclass thiz, jint ctx_id, jstring left_table, jstring right_table,
     jint left_join_col, jint right_join_col,
     jstring join_type_str, jstring join_algorithm_str,
     jstring destination_table) {

  auto ctx = contexts.find(ctx_id)->second;

  auto join_type = join_types.find(jstr_to_str(env, join_type_str))->second;
  auto join_algorithm = join_algorithms.find(jstr_to_str(env, join_algorithm_str))->second;

  auto join_config = cylon::join::config::JoinConfig(join_type, left_join_col, right_join_col, join_algorithm);

  cylon::JoinTables(
      ctx,
      jstr_to_str(env, left_table),
      jstr_to_str(env, right_table),
      join_config,
      jstr_to_str(env, destination_table)
  );
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_Table_nativeDistributedJoin
    (JNIEnv *env, jclass thiz, jint ctx_id, jstring left_table, jstring right_table,
     jint left_join_col, jint right_join_col,
     jstring join_type_str, jstring join_algorithm_str,
     jstring destination_table) {
  auto ctx = contexts.find(ctx_id)->second;

  auto join_type = join_types.find(jstr_to_str(env, join_type_str))->second;
  auto join_algorithm = join_algorithms.find(jstr_to_str(env, join_algorithm_str))->second;

  auto join_config = cylon::join::config::JoinConfig(join_type, left_join_col, right_join_col, join_algorithm);

  cylon::DistributedJoinTables(
      ctx,
      jstr_to_str(env, left_table),
      jstr_to_str(env, right_table),
      join_config,
      jstr_to_str(env, destination_table)
  );
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_Table_nativeLoadCSV
    (JNIEnv *env, jclass thiz, jint ctx_id, jstring path, jstring uuid) {
  auto ctx = contexts.find(ctx_id)->second;
  cylon::Status status = cylon::ReadCSV(ctx, jstr_to_str(env, path),
                                        jstr_to_str(env, uuid));
  if (!status.is_ok()) {
    throwIOException(env, status.get_msg());
  }
}

JNIEXPORT jint JNICALL Java_org_cylondata_cylon_Table_nativeColumnCount
    (JNIEnv *env, jclass thiz, jstring uuid) {
  return cylon::ColumnCount(jstr_to_str(env, uuid));
}

JNIEXPORT jint JNICALL Java_org_cylondata_cylon_Table_nativeRowCount
    (JNIEnv *env, jclass thiz, jstring uuid) {
  return cylon::RowCount(jstr_to_str(env, uuid));
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_Table_print
    (JNIEnv *env, jclass thiz, jstring uuid, jint row1, jint row2, jint col1, jint col2) {
  cylon::Print(jstr_to_str(env, uuid), col1, col2, row1, row2);
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_Table_merge
    (JNIEnv *env, jclass thiz, jint ctx_id, jobjectArray table_ids, jstring merge_tab_id) {
  auto ctx = contexts.find(ctx_id)->second;

  int table_count = env->GetArrayLength(table_ids);

  std::vector<std::string> table_ids_vector;
  for (int i = 0; i < table_count; i++) {
    auto tab_id = (jstring) (env->GetObjectArrayElement(table_ids, i));
    table_ids_vector.push_back(jstr_to_str(env, tab_id));
  }
  cylon::Status status = cylon::Merge(ctx, table_ids_vector, jstr_to_str(env, merge_tab_id));
  std::cout << status.get_code() << std::endl;
  if (!status.is_ok()) {
    throwIOException(env, status.get_msg());
  }
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_Table_clear
    (JNIEnv *env, jclass thiz, jstring table_id) {
  cylon::RemoveTable(jstr_to_str(env, table_id));
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_Table_select
    (JNIEnv *env, jclass thiz, jint ctx_id, jstring table_id, jobject selector, jstring destination_table) {
  auto ctx = contexts.find(ctx_id)->second;

  // selector select method
  jclass selector_cls = env->FindClass("org/cylondata/cylon/ops/Selector");
  jmethodID select_method = env->GetMethodID(selector_cls, "select", "(Lorg/cylondata/cylon/ops/Row;)Z");

  //create a java Row object
  jclass row_cls = env->FindClass("org/cylondata/cylon/ops/Row");
  jmethodID row_cls_constructor = env->GetMethodID(row_cls, "<init>", "()V");
  jfieldID row_id_field = env->GetFieldID(row_cls, "memoryAddress", "J");

  jobject row_obj = env->NewObject(row_cls, row_cls_constructor);
  cylon::Select(ctx,
                jstr_to_str(env, table_id),
                [env, row_obj, row_id_field, selector, select_method](const cylon::Row &row) {
                  // set the current row address
                  env->SetLongField(row_obj, row_id_field, (int64_t) std::addressof(row));

                  // now call the selector
                  return env->CallBooleanMethod(selector, select_method, row_obj);
                },
                jstr_to_str(env, destination_table)
  );
}