package org.twisterx.ops;

public interface Filter<I> {
  boolean filter(I value);
}
