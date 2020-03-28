#include "../include/org_twisterx_join_Join.h"
#include <iostream>

JNIEXPORT void JNICALL
Java_org_twisterx_join_Join_nativeJoin
(JNIEnv
*env,
jobject obj, jobject
buff){
	jbyte *bbuff = (jbyte *) env->GetDirectBufferAddress(buff);
	std::cout << bbuff[0] << std::endl;
}