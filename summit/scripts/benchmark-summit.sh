#! /bin/sh
time jsrun -n 1 python weak_scaling.py
time jsrun -n 8 python weak_scaling.py
#time jsrun -n 10 python weak_scaling.py
#time jsrun -n 16 python weak_scaling.py
#time jsrun -n 32 python weak_scaling.py
#time jsrun -n 64 python weak_scaling.py
#time jsrun -n 72 python weak_scaling.py
#time jsrun -n 84 python weak_scaling.py
