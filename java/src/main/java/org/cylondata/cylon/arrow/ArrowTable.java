package org.cylondata.cylon.arrow;

import io.netty.buffer.ArrowBuf;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.FixedWidthVector;
import org.cylondata.cylon.Clearable;
import org.cylondata.cylon.exception.CylonRuntimeException;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This class provides basic types support to map a java arrow table to cpp arrow table.
 * This class needs to be improved to support complex types
 */
public class ArrowTable implements Clearable {

  // this map will prevent GC while arrow tables are in use in CPP
  private static final Map<String, ArrowTable> ARROW_TABLE_MAP = new HashMap<>();

  private final String uuid;
  private boolean finished;

  // handling memory
  private final List<ArrowBuf> bufferRefs;

  // keep count of references
  private final AtomicInteger references;

  public ArrowTable() {
    this.uuid = UUID.randomUUID().toString();
    this.bufferRefs = new ArrayList<>();
    this.references = new AtomicInteger();
    ArrowTable.createTable(this.uuid);
    ARROW_TABLE_MAP.put(this.uuid, this);
  }

  private void checkFinished() {
    if (this.finished) {
      throw new CylonRuntimeException("This operation is not permitted on a finished table");
    }
  }

  /**
   * Add a column of data. The memory de-allocation and GC prevention will not be handled by Cylon.
   *
   * @param columnName  name of the column
   * @param fieldVector {@link FieldVector} instance
   */
  public void addColumn(String columnName, FieldVector fieldVector) {
    this.addColumn(columnName, fieldVector, false);
  }

  /**
   * Add a column of data
   *
   * @param columnName   name of the column
   * @param fieldVector  {@link FieldVector} instance
   * @param manageMemory If true, Cylon will take care of memory de-allocation for the {@link FieldVector} provided.
   */
  public void addColumn(String columnName, FieldVector fieldVector, boolean manageMemory) {
    ArrowBuf dataBuffer = fieldVector.getDataBuffer();
    ArrowBuf validityBuffer = fieldVector.getValidityBuffer();

    boolean fixedWidth = fieldVector instanceof FixedWidthVector;

    if (!fixedWidth) {
      throw new CylonRuntimeException("Non fixed width vectors are not yet supported.");
    }

    ArrowTable.addColumn(this.uuid,
        columnName,
        fieldVector.getField().getType().getTypeID().getFlatbufID(),
        fieldVector.getValueCount(),
        fieldVector.getNullCount(),
        validityBuffer.memoryAddress(),
        validityBuffer.capacity(),
        dataBuffer.memoryAddress(),
        dataBuffer.capacity()
    );

    if (!manageMemory) {
      dataBuffer.getReferenceManager().retain();
      validityBuffer.getReferenceManager().retain();

      this.bufferRefs.add(dataBuffer);
      this.bufferRefs.add(validityBuffer);
    }
  }

  public void finish() {
    this.checkFinished();
    ArrowTable.finishTable(this.uuid);
    this.finished = true;
  }

  public boolean isFinished() {
    return finished;
  }

  public String getUuid() {
    return uuid;
  }

  private static native void createTable(String tableId);

  private static native void addColumn(String tableId, String columnName, byte type, int valueCount, int nullCount,
                                       long validityAddress, long validitySize,
                                       long dataAddress, long dataSize);

  private static native void finishTable(String tableId);

  public void markReferred() {
    this.references.incrementAndGet();
  }

  @Override
  public void clear() {
    // only proceed if there are no more references
    if (this.references.decrementAndGet() > 0) {
      return;
    }

    // remove from arrow table map
    ARROW_TABLE_MAP.remove(this.uuid);

    // releasing the buffers
    for (ArrowBuf arrowBuf : this.bufferRefs) {
      arrowBuf.getReferenceManager().release();
    }
  }

  @Override
  protected void finalize() throws Throwable {
    this.clear();
  }
}
