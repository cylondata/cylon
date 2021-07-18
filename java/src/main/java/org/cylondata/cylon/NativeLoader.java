 /*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.cylondata.cylon;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.util.logging.Level;
import java.util.logging.Logger;

public class NativeLoader {
  private static final Logger LOG = Logger.getLogger(NativeLoader.class.getName());

  private static String CYLON = "cylon";
  private static String CYLON_JNI = "cylonjni";

  private static boolean loadSuccess = true;

  static {
    loadLibrary(CYLON);
    loadLibrary(CYLON_JNI);
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
