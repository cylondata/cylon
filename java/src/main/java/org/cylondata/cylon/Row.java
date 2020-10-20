package org.cylondata.cylon;

public class Row {

  // this field will be read/write by CPP only
  private long memoryAddress;

  public Row() {
  }

  public native byte getInt8(long column);

  public native short getUInt8(long column);

  public native short getInt16(long column);

  public native int getUInt16(long column);

  public native int getInt32(long column);

  public native long getUInt32(long column);

  public native long getInt64(long column);

  public native long getUInt64(long column);

  public native float getHalfFloat(long column);

  public native float getFloat(long column);

  public native double getDouble(long column);

  public native String getString(long column);
}
