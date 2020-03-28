package org.twisterx.ops;

public interface Mapper<I, O> {
  O map(I cellValue);
}
