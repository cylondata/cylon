from pytwisterx.utils.data import LocalDataLoader

base_path: str = "/home/vibhatha/data/mnist"
train_file_name: str = "mnist_train_small.csv"
test_file_name: str = "mnist_test.csv"

dl = LocalDataLoader(source_dir=base_path, source_files=[train_file_name])

print(dl.source_dir, dl.source_files, dl.file_type, dl.delimiter, dl.loader_type)

dl.load()

for id, dataset in enumerate(dl.dataset):
    print(id, type(dataset))
