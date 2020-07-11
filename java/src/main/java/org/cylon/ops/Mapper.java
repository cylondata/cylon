package org.cylon.ops;

public interface Mapper<I, O> {
  O map(I cellValue);
}
