package org.twisterx.io.arrow;

import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.types.Types;

public class ValueSetter {
  public static void parseAndSet(int index, String value, Types.MinorType type, FieldVector vector) {
    switch (type) {
      case NULL:
        break;
      case STRUCT:
        break;
      case TINYINT:
        break;
      case SMALLINT:
        break;
      case INT:
        int iVal = Integer.parseInt(value);
        ((IntVector) vector).set(index, iVal);
        break;
      case BIGINT:
        break;
      case DATEDAY:
        break;
      case DATEMILLI:
        break;
      case TIMESEC:
        break;
      case TIMEMILLI:
        break;
      case TIMEMICRO:
        break;
      case TIMENANO:
        break;
      case TIMESTAMPSEC:
        break;
      case TIMESTAMPMILLI:
        break;
      case TIMESTAMPMICRO:
        break;
      case TIMESTAMPNANO:
        break;
      case INTERVALDAY:
        break;
      case DURATION:
        break;
      case INTERVALYEAR:
        break;
      case FLOAT4:
        break;
      case FLOAT8:
        break;
      case BIT:
        break;
      case VARCHAR:
        break;
      case VARBINARY:
        break;
      case DECIMAL:
        break;
      case FIXEDSIZEBINARY:
        break;
      case UINT1:
        break;
      case UINT2:
        break;
      case UINT4:
        break;
      case UINT8:
        break;
      case LIST:
        break;
      case FIXED_SIZE_LIST:
        break;
      case UNION:
        break;
      case MAP:
        break;
      case TIMESTAMPSECTZ:
        break;
      case TIMESTAMPMILLITZ:
        break;
      case TIMESTAMPMICROTZ:
        break;
      case TIMESTAMPNANOTZ:
        break;
      case EXTENSIONTYPE:
        break;
    }
  }
}
