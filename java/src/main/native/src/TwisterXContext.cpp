#include "org_twisterx_TwisterXContext.h"
#include "ctx/twisterx_context.h"
#include "net/mpi/mpi_communicator.h"
#include "ConversionUtils.h"

JNIEXPORT void JNICALL Java_org_twisterx_TwisterXContext_nativeInit
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto mpi_config = new twisterx::net::MPIConfig();
  auto ctx = twisterx::TwisterXContext::InitDistributed(mpi_config);
  contexts.insert(std::pair<int32_t, twisterx::TwisterXContext *>(ctx_id, ctx));
}

JNIEXPORT void JNICALL Java_org_twisterx_TwisterXContext_barrier
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto ctx = contexts.find(ctx_id)->second;
  ctx->Barrier();
}

JNIEXPORT void JNICALL Java_org_twisterx_TwisterXContext_finalize
    (JNIEnv *env, jclass obj, jint ctx_id) {
  auto ctx = contexts.find(ctx_id)->second;
  ctx->Finalize();
}