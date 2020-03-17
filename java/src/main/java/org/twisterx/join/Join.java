package org.twisterx.join;

import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowFileWriter;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

import java.nio.ByteBuffer;
import java.util.Arrays;

public class Join {

  public void join() {


  }

  public native void nativeJoin(ByteBuffer... buffer);


  public static void main(String[] args) {
    //System.setProperty("prefix", "~/Code/twisterx/java/target/native-deps/");
    //NativeLoader.load();

    System.load("/home/chathura/Code/twisterx/cpp/build/arrow/build/release/libarrow.so.16.0.0");
    System.load("/home/chathura/Code/twisterx/java/target/native-deps/amd64/Linux/libtwisterx.so");
    System.load("/home/chathura/Code/twisterx/java/target/native-deps/amd64/Linux/libtwisterxjni.so");
    Field intField = new Field("int", FieldType.nullable(new ArrowType.Int(32, true)), null);
    Schema schema = new Schema(Arrays.asList(intField));

    RootAllocator rootAllocator = new RootAllocator(Integer.MAX_VALUE);
    VectorSchemaRoot root = VectorSchemaRoot.create(schema, rootAllocator);
    IntVector iv = new IntVector("ints", rootAllocator);
    iv.allocateNew();
    iv.set(0, 123);

    root.addVector(0, iv);
    root.close();
    System.out.println(root.contentToTSVString());



    new Join().nativeJoin(iv.getDataBuffer().nioBuffer());
  }
}


