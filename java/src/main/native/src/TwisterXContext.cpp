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

#include "org_cylondata_cylon_CylonContext.h"
#include "ctx/cylon_context.hpp"
#include "net/mpi/mpi_communicator.hpp"
#include "ConversionUtils.h"

JNIEXPORT void JNICALL Java_org_cylondata_cylon_CylonContext_nativeInit
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
  contexts.insert(std::pair<int32_t, cylon::CylonContext *>(ctx_id, ctx));
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_CylonContext_barrier
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto ctx = contexts.find(ctx_id)->second;
  ctx->Barrier();
}

JNIEXPORT void JNICALL Java_org_cylondata_cylon_CylonContext_finalize
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto ctx = contexts.find(ctx_id)->second;
  ctx->Finalize();
}

JNIEXPORT jint JNICALL Java_org_cylondata_cylon_CylonContext_getWorldSize
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto ctx = contexts.find(ctx_id)->second;
  return ctx->GetWorldSize();
}

JNIEXPORT jint JNICALL Java_org_cylondata_cylon_CylonContext_getRank
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto ctx = contexts.find(ctx_id)->second;
  return ctx->GetRank();
}