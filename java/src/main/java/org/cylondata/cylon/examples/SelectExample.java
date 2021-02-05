package org.cylondata.cylon.examples;

import org.cylondata.cylon.CylonContext;
import org.cylondata.cylon.Table;

public class SelectExample {
  public static void main(String[] args) {
    String tablePath = args[0];

    // initialize Cylon context
    CylonContext ctx = CylonContext.init();

    // create a table from csv
    Table srcTable = Table.fromCSV(ctx, tablePath);

    final long somethingOutside = 7;

    // applying select operation
    Table select = srcTable.select((row) -> row.getInt64(0) == somethingOutside);

    // print the table to console
    select.print();

    // finalizing Cylon context
    ctx.finalizeCtx();
  }
}
