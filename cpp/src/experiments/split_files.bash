#!/bin/bash

# ex: ./split_files.bash /tmp/csv1.csv csv1 /tmp/cylon/strong/

splitCsv() {
    HEADER=$(head -1 "$1")
    if [ -n "$2" ]; then
        CHUNK=$2
    else 
        CHUNK=1000
    fi
    tail -n +2 "$1" | split -d -a 3 --additional-suffix .csv -l "$CHUNK" - "$3_"
#     for i in "$3"_*; do
#         sed -i -e "1i$HEADER" "$i"
#     done
}


DEST_BASE=$3
# LINES=1000000000
LINES=$(cat "$1" | wc -l )


echo "$1 $2 $3 $LINES"

# for w in 2 4 8 16 32 64 128 160 
# for w in 256 512 
for w in 200
do
    dest="$DEST_BASE/$w"
    mkdir -p $dest 
    
    lines=$(( (LINES - 1)/w ))
    
    if ((lines > 0)); then
        echo "splitting for $w $lines $dest"

        splitCsv "$1" "$lines" "$dest/$2"

        echo "splitting $w done!"
    fi
done 