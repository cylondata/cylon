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

package org.cylondata.cylon;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * CylonContext can be used to initialize communication and load native Cylon
 * dependencies into the java project
 */
public class CylonContext {

  private static AtomicInteger ctxIdGenerator = new AtomicInteger();

  private int ctxId;

  private CylonContext() {
    this.ctxId = ctxIdGenerator.getAndIncrement();
  }

  /**
   * Loads Cylon native dependencies and initializes Cylon communication
   *
   * @return and instance of {@link CylonContext}
   */
  public static CylonContext init() {
    boolean loaded = NativeLoader.load();

    if (!loaded) {
      throw new RuntimeException("Failed to load cylon native libraries");
    }

    CylonContext ctx = new CylonContext();
    CylonContext.nativeInit(ctx.getCtxId());
    return ctx;
  }

  /**
   * Synchronizes all the worker processes across the cluster on a barrier
   */
  public void barrier() {
    CylonContext.barrier(this.ctxId);
  }

  /**
   * Close the communication chancels and do the final memory cleanup
   */
  public void finalizeCtx() {
    CylonContext.finalize(this.ctxId);
  }

  /**
   * Retrieve the number of the workers involved in this Cylon Job
   *
   * @return size of the world
   */
  public int getWorldSize() {
    return CylonContext.getWorldSize(this.ctxId);
  }

  /**
   * Retrieve the Rank(ID) of this worker
   *
   * @return rank of this worker
   */
  public int getRank() {
    return CylonContext.getRank(this.ctxId);
  }

  public static native int getWorldSize(int ctxId);

  public static native int getRank(int ctxId);

  public static native void finalize(int ctxId);

  public static native void barrier(int ctxId);

  private static native void nativeInit(int ctxId);

  /**
   * @return ID of this context.
   */
  public int getCtxId() {
    return ctxId;
  }
}
