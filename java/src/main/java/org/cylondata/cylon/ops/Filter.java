package org.cylondata.cylon.ops;

public interface Filter<I> {
  boolean filter(I value);
}
