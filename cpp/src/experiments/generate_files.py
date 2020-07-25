import argparse
import os
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='generate random data')
parser.add_argument('-w', dest='world', type=int, nargs='+', help='world sizes',
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 160])
parser.add_argument('-t', dest='threads', type=int, help='file writing threads', default=64)
parser.add_argument('--wr', dest='weak_rows', type=float, nargs='+', help='weak rows',
                    default=[])
parser.add_argument('--sr', dest='strong_rows', type=float, nargs='+', help='strong rows',
                    default=[])
parser.add_argument('-b', dest='base_dir', type=str, help='base dir', default="~/temp/cylon/")
parser.add_argument('-c', dest='cols', type=int, help='columns', default=2)


args = parser.parse_args()
args = vars(args)
print(f"file gen args {args}", flush=True)

world_sizes = args['world']
file_gen_threads = args['threads']
base_dir = args["base_dir"]

print(f"base dir {base_dir}", flush=True)
os.system(f"mkdir -p {base_dir}")

csvs = ["BASE/csv1_RANK.csv", "BASE/csv2_RANK.csv"]
cols = args['cols']
key_duplication_ratio = 0.99  # on avg there will be rows/key_range_ratio num of duplicate keys


def generate_files(b, _rank, _i, _krange):
    # print(f"generating files for {_i} {_rank}")
    for _f in csvs:
        out_f = _f.replace('RANK', str(_rank)).replace('BASE', b)
        if not os.path.exists(out_f):
            os.system(
                f"python generate_csv.py -o {out_f} -r {_i} -c {cols} --krange 0 {_krange[1]}")


work = []

# weak scaling
row_cases = [int(ii * 1000000) for ii in args['weak_rows']]

for i in row_cases:
    for w in world_sizes:
        krange = (0, int(i * key_duplication_ratio * w))

        out_d = f"{base_dir}/weak/{i}/{w}/"
        os.system(f"mkdir -p {out_d}")

        # generate 2 cvs for world size
        print(f"##### generating files of rows {i} {w} {out_d}!", flush=True)
        for rank in range(w):
            work.append((out_d, rank, i, krange))

# strong scaling
row_cases = [int(ii * 1000000) for ii in args['strong_rows']]

for i in row_cases:
    for w in world_sizes:
        krange = (0, int(i * key_duplication_ratio))
        out_d = f"{base_dir}/strong/{i}/{w}/"
        os.system(f"mkdir -p {out_d}")

        # generate 2 cvs for world size
        print(f"##### generating files of rows {i} {w}!", flush=True)
        for rank in range(w):
            work.append((out_d, rank, int(i / w), krange))

print("work:", work, flush=True)

p = Pool(file_gen_threads)
p.starmap(generate_files, work)
p.close()
p.join()


# os.system(f"tar czf twx_input.tar {base_dir} ")

print("DONE!", flush=True)

