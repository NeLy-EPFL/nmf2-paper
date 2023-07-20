data_dir="outputs"
temp_file=videos_list.txt

for trial in $(seq 0 9); do
    echo "file '$data_dir/odor_taxis_$trial.mp4'" >> $temp_file
done

ffmpeg -f concat -safe 0 -i $temp_file -c copy $data_dir/odor_taxis_all_trials.mp4

rm $temp_file
