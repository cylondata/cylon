package org.cylondata.cylon.ops;

import org.cylondata.cylon.Row;

public interface Selector {
  boolean select(Row row);
}
