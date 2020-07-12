package org.cylon.arrow;

import com.google.flatbuffers.FlatBufferBuilder;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.cylon.NativeLoader;

import java.util.ArrayList;
import java.util.List;

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

    List<Field> arrowFields = new ArrayList<>();
    arrowFields.add(Field.nullable("col1", new ArrowType.Int(8, true)));

    Schema schema = new Schema(arrowFields);
    schema.toByteArray();
    System.out.println(schema.toJson());


    ArrowTable.addColumn("", 0, intVector.getDataBufferAddress(), 200 * 4);
  }

  private static native void createTable(String tableId);

  private static native void addColumn(String tableId, int type, long address, long size);

  private static native void finishTable(String tableId);
}
