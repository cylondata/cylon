#! /bin/sh
squeue -u $USER | awk '{print $1}' | tail -n+2 | xargs scancel
