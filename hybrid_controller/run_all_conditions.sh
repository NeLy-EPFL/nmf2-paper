#!/bin/bash

parallelize=true
# ...do something interesting...
if [ "$parallelize" = true ] ; then
    echo "Running all arenas with adhesion [PARALLEL]"
    python generate_datapts.py --arena flat --adhesion --n_procs 4 --n_exp 10
    echo "1/4"
    python generate_datapts.py --arena gapped --adhesion --n_procs 4 --n_exp 10
    echo "2/4"
    python generate_datapts.py --arena blocks --adhesion --n_procs 4 --n_exp 10
    echo "3/4"
    python generate_datapts.py --arena mixed --adhesion  --n_procs 4 --n_exp 10
    echo "4/4"

<<com
    echo "Running all arenas without adhesion"
    python generate_datapts.py --arena flat --n_procs 2
    echo "1/4"
    python generate_datapts.py --arena gapped --n_procs 2
    echo "2/4"
    python generate_datapts.py --arena blocks --n_procs 2
    echo "3/4"
    python generate_datapts.py --arena mixed --n_procs 2
    echo "4/4"
com
else
    echo "Running all arenas without adhesion [NOT PARALLEL]"
    python generate_datapts.py --arena flat
    echo "1/4"
    python generate_datapts.py --arena gapped
    echo "2/4"
    python generate_datapts.py --arena blocks   
    echo "3/4"
    python generate_datapts.py --arena mixed  
    echo "4/4"

<<com
    echo "Running all arenas with adhesion"
    python generate_datapts.py --arena flat --adhesion 
    echo "1/4"
    python generate_datapts.py --arena gapped --adhesion 
    echo "2/4"
    python generate_datapts.py --arena blocks --adhesion
    echo "3/4"
    python generate_datapts.py --arena mixed --adhesion
    echo "4/4"
com
fi

echo "Merge videos"
python merge_videos.py

echo "Generate figure"
python generate_figure.py

echo "Done"
