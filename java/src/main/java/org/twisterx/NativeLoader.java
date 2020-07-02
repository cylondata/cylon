package org.twisterx;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.util.logging.Level;
import java.util.logging.Logger;

public class NativeLoader {
  private static final Logger LOG = Logger.getLogger(NativeLoader.class.getName());

  private static String TWISTERX = "twisterx";
  private static String TWISTERX_JNI = "twisterxjni";

  private static boolean loadSuccess = true;

  static {
    loadLibrary(TWISTERX);
    loadLibrary(TWISTERX_JNI);
  }


  public static boolean load() {
    return loadSuccess;
  }

  private static void loadLibrary(String file) {
    try {
      String os = System.getProperty("os.name");
      String arch = System.getProperty("os.arch");
      String prefix = System.getProperty("prefix", "") + arch + "/" + os;

      String path = prefix + "/" + System.mapLibraryName(file);
      System.out.println("Loading libraries from " + path);
      URL resource = NativeLoader.class.getClassLoader().getResource(path);

      if (resource == null) {
        LOG.log(Level.SEVERE, "Cannot find the file - " + path);
        loadSuccess = false;
        return;
      }

      InputStream instream = resource.openStream();
      File tempFile = File.createTempFile(file, ".so");
      FileOutputStream outstream = new FileOutputStream(tempFile);

      byte[] buffer = new byte[1024];

      int length;
      while ((length = instream.read(buffer)) > 0) {
        outstream.write(buffer, 0, length);
      }

      instream.close();
      outstream.close();

      tempFile.deleteOnExit();
      System.load(tempFile.getAbsolutePath());
      tempFile.delete();
    } catch (Throwable t) {
      loadSuccess = false;
      LOG.log(Level.SEVERE, "Failed to load the library ", t);
    }
  }
}
