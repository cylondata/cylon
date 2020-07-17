---
id: java
title: Java API
---

With Cylon's java binding(JCylon), Java programmers can achieve the same performance of a native C++ Cylon code and minimize downsides of java codes such as garbage collection overheads.

## Cylon Java Binding Example

You can start writing a cylon application by including the JCylon dependency as below in your pom.xml.

```xml
<dependency>
    <groupId>org.cylondata</groupId>
    <artifactId>cylon</artifactId>
    <version>${cylon.version}</version>
</dependency>
```

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

## Running a Cylon Java Example

Once the example is built into a JAR, it can be run as follows.

```bash
mpirun -np 4 <CYLON_HOME>/bin/join_example /path/to/csv1 /path/to/csv2
```

## JCylon Docs

Use blow link to navigate to the JCylon javadocs.

<a href="../javadocs/index.html" target="_blank">Javadocs</a>