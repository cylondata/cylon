##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##

'''
Run test:
>> pytest -q python/test/test_build_arrow.py
'''



import os
import sys
import pyarrow


def test_build():
    pyarrow_location = os.path.dirname(pyarrow.__file__)
    # For now, assume that we build against bundled pyarrow releases.
    pyarrow_include_dir = os.path.join(pyarrow_location, 'include')

    print(pyarrow_location)
    print(pyarrow_include_dir)

    if sys.platform == "darwin":
        print("Darwin")
    else:
        print(sys.platform)