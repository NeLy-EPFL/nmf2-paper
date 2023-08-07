#!/bin/bash

parallelize=true
# ...do something interesting...
if [ "$parallelize" = true ] ; then
    echo "Running all arenas with adhesion [PARALLEL]"
    python generate_datapts.py --arena flat --adhesion --parallel 
    echo "1/4"
    python generate_datapts.py --arena gapped --adhesion --parallel
    echo "2/4"
    python generate_datapts.py --arena blocks --adhesion --parallel 
    echo "3/4"
    python generate_datapts.py --arena mixed --adhesion  --parallel
    echo "4/4"

    echo "Running all arenas without adhesion"
    python generate_datapts.py --arena flat --parallel
    echo "1/4"
    python generate_datapts.py --arena gapped --parallel
    echo "2/4"
    python generate_datapts.py --arena blocks --parallel
    echo "3/4"
    python generate_datapts.py --arena mixed --parallel
    echo "4/4"
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

    echo "Running all arenas with adhesion"
    python generate_datapts.py --arena flat --adhesion 
    echo "1/4"
    python generate_datapts.py --arena gapped --adhesion 
    echo "2/4"
    python generate_datapts.py --arena blocks --adhesion
    echo "3/4"
    python generate_datapts.py --arena mixed --adhesion
    echo "4/4"
fi

echo "Merge videos"
python merge_videos.py

echo "Generate figure"
python generate_figure.py

echo "Done"
