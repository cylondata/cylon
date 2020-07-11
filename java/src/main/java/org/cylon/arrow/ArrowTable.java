package org.cylon.arrow;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;

public class ArrowTable {
  public static void main(String[] args) {
    RootAllocator rootAllocator = new RootAllocator();
    IntVector intVector = new IntVector("col1", rootAllocator);
    intVector.allocateNew(100);
    intVector.set(0, 1);
    System.out.println(intVector.get(0));
    System.out.println(intVector.getDataBuffer().memoryAddress());

    System.out.println(intVector.getDataBuffer().nioBuffer());
  }

  private static native void createTable(String tableId);

  private static native void addColumn(String tableId, int type, long address);

  private static native void finishTable(String tableId);
}
