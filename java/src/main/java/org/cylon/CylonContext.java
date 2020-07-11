package org.cylon;

import java.util.concurrent.atomic.AtomicInteger;

public class CylonContext {

  private static AtomicInteger ctxIdGenerator = new AtomicInteger();

  private int ctxId;

  private CylonContext() {
    this.ctxId = ctxIdGenerator.getAndIncrement();
  }

  public static CylonContext init() {
    boolean loaded = NativeLoader.load();

    if (!loaded) {
      throw new RuntimeException("Failed to load cylon native libraries");
    }

    CylonContext ctx = new CylonContext();
    CylonContext.nativeInit(ctx.getCtxId());
    return ctx;
  }

  public void barrier() {
    CylonContext.barrier(this.ctxId);
  }

  public void finalizeCtx() {
    CylonContext.finalize(this.ctxId);
  }

  public int getWorldSize() {
    return CylonContext.getWorldSize(this.ctxId);
  }

  public int getRank() {
    return CylonContext.getRank(this.ctxId);
  }

  public static native int getWorldSize(int ctxId);

  public static native int getRank(int ctxId);

  public static native void finalize(int ctxId);

  public static native void barrier(int ctxId);

  private static native void nativeInit(int ctxId);

  public int getCtxId() {
    return ctxId;
  }
}
