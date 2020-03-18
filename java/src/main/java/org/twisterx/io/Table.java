package org.twisterx.io;

import org.twisterx.NativeLoader;

import java.io.IOException;
import java.util.UUID;

public class Table {
  
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

  public Table join(Table rightTable, int leftIdx, int rightIdx) {
    String uuid = UUID.randomUUID().toString();
    Table.nativeJoin(this.tableId, rightTable.tableId, leftIdx, rightIdx, uuid);
    return new Table(uuid);
  }

  // native methods

  /**
   * @param left        id of the left table
   * @param right       id of the right table
   * @param tab1Index   left join column index
   * @param tab2Index   right join column index
   * @param destination destination table id
   */
  public static native void nativeJoin(String left, String right, int tab1Index, int tab2Index,
                                       String destination);

  private static native int nativeColumnCount(String tableId);

  private static native int nativeRowCount(String tableId);

  public static native void nativeLoadCSV(String path, String id);
  // end of native methods

  public static void main(String[] args) throws IOException {
    NativeLoader.load();

    Table left = Table.fromCSV("/tmp/csv.csv");
    Table right = Table.fromCSV("/tmp/csv.csv");
    Table joined = left.join(right, 0, 0);

    System.out.println(joined.getColumnCount());
    System.out.println(joined.getRowCount());
  }
}
