import os
import sys
import pyarrow

pyarrow_location = os.path.dirname(pyarrow.__file__)
# For now, assume that we build against bundled pyarrow releases.
pyarrow_include_dir = os.path.join(pyarrow_location, 'include')


print(pyarrow_location)
print(pyarrow_include_dir)

if sys.platform == "darwin":
    print("Darwin")
else:
    print(sys.platform)