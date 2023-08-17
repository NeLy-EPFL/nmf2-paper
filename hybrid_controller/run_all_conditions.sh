#!/bin/bash

n_procs=1
n_exp=20

for arena in flat gapped blocks mixed; do
    echo "Running simulations for $arena arena..."
    python generate_datapts.py --arena $arena --adhesion --n_procs $n_procs --n_exp $n_exp &
done

echo wait $(jobs -p)
wait $(jobs -p)

#echo "Merge videos"
#python merge_videos.py

#echo "Generate figure"
#python generate_figure.py

echo "Done"
