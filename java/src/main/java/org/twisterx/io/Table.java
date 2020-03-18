package org.twisterx.io;

import org.apache.arrow.vector.VectorSchemaRoot;
import org.twisterx.NativeLoader;

import java.io.IOException;
import java.util.UUID;

public class Table {

  private VectorSchemaRoot vectorSchemaRoot;
  private String tableId;

  private Table(String tableId) {
    this.tableId = tableId;
  }

  public static Table fromCSV(String path) {
    String uuid = UUID.randomUUID().toString();
    Table.nativeLoadCSV(path, uuid);
    return new Table(uuid);
  }

  public int getColumnCount() {
    return Table.nativeColumnCount(this.tableId);
  }

  public int getRowCount() {
    return Table.nativeRowCount(this.tableId);
  }

  private String getTableId() {
    return tableId;
  }

  // native methods
  public static native void join(String left, String right, int tab1Index, int tab2Index);

  private static native int nativeColumnCount(String tableId);

  private static native int nativeRowCount(String tableId);

  public static native void nativeLoadCSV(String path, String id);
  // end of native methods

  public static void main(String[] args) throws IOException {
    NativeLoader.load();

    Table table = Table.fromCSV("/tmp/csv.csv");
    System.out.println(table.getColumnCount());
    System.out.println(table.getColumnCount());
  }
}
