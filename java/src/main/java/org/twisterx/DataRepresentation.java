package org.twisterx;

@SuppressWarnings("unused")
public class DataRepresentation {

  private String id;

  public DataRepresentation(String id) {
    this.id = id;
  }

  protected static UnsupportedOperationException unSupportedException() {
    return new UnsupportedOperationException("This operation is not supported yet");
  }

  /**
   * Returns the unique ID assigned for this {@link DataRepresentation}.
   * This can be useful when cross API scripting is required.
   *
   * @return ID
   */
  public String getId() {
    return id;
  }
}
