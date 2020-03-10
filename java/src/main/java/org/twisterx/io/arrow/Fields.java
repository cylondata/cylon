package org.twisterx.io.arrow;

import org.apache.arrow.vector.types.Types;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;

public class Fields {
  public static Field createDefaultField(String name, Types.MinorType type) {
    return new Field(name, FieldType.nullable(type.getType()), null);
  }
}
