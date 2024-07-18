ffmpeg \
    -y
    -i outputs/controller_comparison.mp4 \
    -vcodec libx264 \
    -crf 34 \
    outputs/controller_comparison_small.mp4
