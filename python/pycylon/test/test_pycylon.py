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
running test case
>>  pytest -q python/pycylon/test/test_pycylon.py
'''

def test_cylon_install():
    no_cylon = False
    try:
        import pycylon
    except ImportError:
        no_cylon = True

    if no_cylon:
        print("No PyCylon installation found!")
    else:
        print("PyCylon Installed!")
