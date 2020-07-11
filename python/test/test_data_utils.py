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

from pycylon.util.data.DataManager import LocalDataLoader

base_path: str = "/home/vibhatha/data/mnist"
train_file_name: str = "mnist_train_small.csv"
test_file_name: str = "mnist_test.csv"

dl = LocalDataLoader(source_dir=base_path, source_files=[train_file_name])

print(dl.source_dir, dl.source_files, dl.file_type, dl.delimiter, dl.loader_type)

dl.load()

for id, dataset in enumerate(dl.dataset):
    print(id, type(dataset), dataset.to_pandas().to_numpy().shape)
