import redis
import argparse
import os

def environ_or_required(key):
    return (
        {'default': os.environ.get(key)} if os.environ.get(key)
        else {'required': True}
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clear redis db")
    parser.add_argument("-r", dest='redis_host', type=str, help="redis address, default to 127.0.0.1",
                        **environ_or_required('REDIS_HOST')) #127.0.0.1
    parser.add_argument("-p1", dest='redis_port', type=int, help="name of redis port", **environ_or_required('REDIS_PORT')) #6379



    args = vars(parser.parse_args())

    r = redis.Redis(host=args['redis_host'], port=args['redis_port'], db=0)
    if r is not None:
        r.flushdb()
        print("flushed redisDB")
    else:
        print("warn - could not flush db")




