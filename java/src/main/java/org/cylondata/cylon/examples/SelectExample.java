package org.cylondata.cylon.examples;

import org.cylondata.cylon.CylonContext;
import org.cylondata.cylon.Table;

public class SelectExample {
  public static void main(String[] args) {
    String tablePath = args[0];

    CylonContext ctx = CylonContext.init();

    Table srcTable = Table.fromCSV(ctx, tablePath);

    final long somethingOutside = 4;

    Table select = srcTable.select((row) -> {
      return row.getInt64(0) == somethingOutside;
    });

    select.print();
    ctx.finalizeCtx();
  }
}
