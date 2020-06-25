package org.twisterx;

import java.util.concurrent.atomic.AtomicInteger;

public class TwisterXContext {

  private static AtomicInteger ctxIdGenerator = new AtomicInteger();

  private int ctxId;

  private TwisterXContext() {
    this.ctxId = ctxIdGenerator.getAndIncrement();
  }

  public static TwisterXContext init() {
    boolean loaded = NativeLoader.load();

    if (!loaded) {
      throw new RuntimeException("Failed to load twisterx native libraries");
    }

    TwisterXContext ctx = new TwisterXContext();
    TwisterXContext.nativeInit(ctx.getCtxId());
    return ctx;
  }

  public void barrier() {
    TwisterXContext.barrier(this.ctxId);
  }

  public void finalizeCtx() {
    TwisterXContext.finalize(this.ctxId);
  }

  public static native void finalize(int ctxId);

  public static native void barrier(int ctxId);

  private static native void nativeInit(int ctxId);

  public int getCtxId() {
    return ctxId;
  }
}
