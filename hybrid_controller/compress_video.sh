ffmpeg \
    -i outputs/controller_comparison.mp4 \
    -b:v 1000k \
    -b:a 128k \
    -vf "scale=-1:720" \
    outputs/controller_comparison_small.mp4
