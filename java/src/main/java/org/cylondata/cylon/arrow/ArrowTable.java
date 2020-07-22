package org.cylondata.cylon.arrow;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.cylondata.cylon.NativeLoader;
import org.cylondata.cylon.exception.CylonRuntimeException;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * This class provides basic types support to map a java arrow table to cpp arrow table.
 * This class needs to be improved to support complex types
 */
public class ArrowTable {

  private final String uuid;
  private Schema schema;
  private boolean finished;

  public static void main(String[] args) {

    NativeLoader.load();

    RootAllocator rootAllocator = new RootAllocator();
    IntVector intVector = new IntVector("col1", rootAllocator);
    intVector.allocateNew(200);

    Float8Vector float8Vector = new Float8Vector("col2", rootAllocator);
    for (int i = 0; i < 200; i++) {
      float8Vector.setSafe(i, i);
    }


    List<Field> arrowFields = new ArrayList<>();
    arrowFields.add(Field.nullable("col1", new ArrowType.Int(8, true)));
    arrowFields.add(Field.nullable("col2", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)));


    Schema schema = new Schema(arrowFields);
    schema.toByteArray();

    ArrowTable arrowTable = new ArrowTable(schema);
    arrowTable.addColumn("col1", intVector);
    arrowTable.addColumn("col2", float8Vector);
    arrowTable.finish();
  }

  public ArrowTable(Schema schema) {
    this.schema = schema;
    this.uuid = UUID.randomUUID().toString();
    ArrowTable.createTable(this.uuid);
  }

  private void checkFinished() {
    if (this.finished) {
      throw new CylonRuntimeException("This operation is not permitted on a finished table");
    }
  }

  public void addColumn(String columnName, FieldVector fieldVector) {
    ArrowTable.addColumn(this.uuid,
        columnName,
        this.schema.findField(columnName).getType().getTypeID().getFlatbufID(),
        fieldVector.getDataBufferAddress(),
        fieldVector.getBufferSize()
    );
  }

  public void finish() {
    this.checkFinished();
    ArrowTable.finishTable(this.uuid);
    this.finished = true;
  }

  public String getUuid() {
    return uuid;
  }

  private static native void createTable(String tableId);

  private static native void addColumn(String tableId, String columnName, int type, long address, long size);

  private static native void finishTable(String tableId);
}
