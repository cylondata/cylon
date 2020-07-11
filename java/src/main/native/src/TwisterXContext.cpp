#include "org_cylon_CylonContext.h"
#include "ctx/cylon_context.hpp"
#include "net/mpi/mpi_communicator.hpp"
#include "ConversionUtils.h"

JNIEXPORT void JNICALL Java_org_cylon_CylonContext_nativeInit
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
  contexts.insert(std::pair<int32_t, cylon::CylonContext *>(ctx_id, ctx));
}

JNIEXPORT void JNICALL Java_org_cylon_CylonContext_barrier
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto ctx = contexts.find(ctx_id)->second;
  ctx->Barrier();
}

JNIEXPORT void JNICALL Java_org_cylon_CylonContext_finalize
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto ctx = contexts.find(ctx_id)->second;
  ctx->Finalize();
}

JNIEXPORT jint JNICALL Java_org_cylon_CylonContext_getWorldSize
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto ctx = contexts.find(ctx_id)->second;
  return ctx->GetWorldSize();
}

JNIEXPORT jint JNICALL Java_org_cylon_CylonContext_getRank
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto ctx = contexts.find(ctx_id)->second;
  return ctx->GetRank();
}