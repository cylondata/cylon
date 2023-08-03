import os
import argparse
import subprocess
import redis

def main(world_size: int, redis_addr: str, executable_name: str):
    host, port = redis_addr.split(':')
    r = redis.Redis(host, int(port), db=0)
    r.flushall()
    d = dict(os.environ)
    d["CYLON_UCX_OOB_WORLD_SIZE"] = str(world_size)
    d["CYLON_UCX_OOB_REDIS_ADDR"] = redis_addr
    children = []
    for _ in range(world_size):
        children.append(subprocess.Popen(executable_name, env=d))

    for child in children:
        child.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', "-n", type=int, help="world size")
    parser.add_argument("--redis_addr", "-r", type=str, help="redis address, default to 127.0.0.1:6379", default="127.0.0.1:6379")
    parser.add_argument("--execute", "-e", type=str, help="name of executable")
    args = parser.parse_args()
    main(args.world_size, args.redis_addr, args.execute)
