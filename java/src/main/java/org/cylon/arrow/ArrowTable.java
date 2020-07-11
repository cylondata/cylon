package org.cylon.arrow;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.cylon.NativeLoader;

public class ArrowTable {
  public static void main(String[] args) {

    NativeLoader.load();

    RootAllocator rootAllocator = new RootAllocator();
    IntVector intVector = new IntVector("col1", rootAllocator);
    intVector.allocateNew(200);
    //intVector.setValueCount(200);
    for (int i = 0; i < 200; i++) {
      intVector.setSafe(i, i);
    }
    intVector.setValueCount(200);

    ArrowTable.addColumn("", 0, intVector.getDataBufferAddress(), 200 * 4);
  }

  private static native void createTable(String tableId);

  private static native void addColumn(String tableId, int type, long address, long size);

  private static native void finishTable(String tableId);
}
