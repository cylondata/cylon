
#include <unordered_map>
#include "../include/org_twisterx_io_Table.h"
#include "io/table_api.h";

JNIEXPORT void JNICALL Java_org_twisterx_io_Table_join
    (JNIEnv *env, jclass claz, jobject thiz, jint left_col, jint right_col) {

}

JNIEXPORT void JNICALL Java_org_twisterx_io_Table_nativeLoadCSV
    (JNIEnv *env, jclass thiz, jstring path, jstring uuid) {
  jboolean is_copy;
  std::string path_str = std::string(env->GetStringUTFChars(path, &is_copy));
  std::string uuid_str = std::string(env->GetStringUTFChars(uuid, &is_copy));
  int result = twisterx::io::read_csv(path_str, uuid_str);
}

JNIEXPORT jint JNICALL Java_org_twisterx_io_Table_nativeColumnCount
    (JNIEnv *env, jclass thiz, jstring uuid){

}