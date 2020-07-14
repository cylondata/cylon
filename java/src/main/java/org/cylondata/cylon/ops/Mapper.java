package org.cylondata.cylon.ops;

public interface Mapper<I, O> {
  O map(I cellValue);
}
