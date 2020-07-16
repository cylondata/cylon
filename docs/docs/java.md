---
id: java
title: Java API
hide_table_of_contents: true
---

With Cylon's java binding, Java programmers can achieve the same performance of a native C++ Cylon code and minimize downsides of java codes such as garbage collection overheads.

Given below is an example written with Cylon's java binding to perform the Select operation.


```java
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

    final long somethingOutside = 4;

    // applying select operation
    Table select = srcTable.select((row) -> row.getInt64(0) == somethingOutside);

    // print the table to console
    select.print();

    // finalizing Cylon context
    ctx.finalizeCtx();
  }
}
```

Refer javadoc below to explore the capabilities of Cylon's Java binding.

<iframe src="../javadocs/index.html" width="100%" scrolling="no" frameBorder="0" height="800">

</iframe>