#!/bin/bash
video_dir="_outputs/VIDEOS"
figure_dir="_outputs/FIGURES"
data_dir="_outputs/DATA"

mkdir -p $video_dir
mkdir -p $figure_dir
mkdir -p $data_dir

vid2="${video_dir}/video_2_single_step.mp4"
vid3="${video_dir}/video_3_force_visualization.mp4"
vid4="${video_dir}/video_4_climbing.mp4"
vid5="${video_dir}/video_5_cpg_controller.mp4"
vid6="${video_dir}/video_6_rule_based_controller.mp4"
vid7="${video_dir}/video_7_hybrid_controller.mp4"
vid8="${video_dir}/video_8_controller_comparison_small.mp4"
vid9="${video_dir}/video_9_visual_taxis.mp4"
vid10="${video_dir}/video_10_odor_taxis.mp4"
vid13="${video_dir}/video_13_multimodal_navigation_example.mp4"

edfig2="${figure_dir}/edfig2_preprogrammed_stepping.pdf"

edfig3a="${figure_dir}/edfig3a_birdeye_view.png"
edfig3b="${figure_dir}/edfig3b_birdeye_zoom_view.png"
edfig3ci="${figure_dir}/edfig3ci_raw_img_L.png"
edfig3cii="${figure_dir}/edfig3cii_raw_img_R.png"
edfig3di="${figure_dir}/edfig3di_corrected_img_L.png"
edfig3dii="${figure_dir}/edfig3dii_corrected_img_R.png"
edfig3ei="${figure_dir}/edfig3ei_human_view_L.png"
edfig3eii="${figure_dir}/edfig3eii_human_view_R.png"

edfig6="${figure_dir}/edfig6_vison_model_rl.pdf"

fig1b="${figure_dir}/fig1b_schematics_env_overview.png"

fig2b="${figure_dir}/fig2b_locomotion_climbing.pdf"
fig2d="${figure_dir}/fig2d_locomotion_terrains.pdf"
fig2c="${figure_dir}/fig2c_locomotion_critical_slope.pdf"
fig2g="${figure_dir}/fig2g_locomotion_controller_comparison.pdf"

fig3bi="${figure_dir}/fig3bi_sensory_vision_sim.pdf"
fig3bii="${figure_dir}/fig3bii_sensory_behind_fly_view.png"
fig3c="${figure_dir}/fig3c_sensory_visual_taxis.pdf"
fig3d="${figure_dir}/fig3d_sensory_odor_taxis.pdf"

fig5b="${figure_dir}/fig5b_integration_trajectory.pdf"
fig5c="${figure_dir}/fig5c_integration_trajectories.pdf"

data_fig2c="${data_dir}/fig2c_locomotion_critical_slope.csv"
data_fig2g="${data_dir}/fig2g_locomotion_controller_comparison.csv"
data_ed2fig2="${data_dir}/edfig2_preprogrammed_stepping.csv"
data_edfig6c="${data_dir}/edfig6c_vison_model_rl_direction.csv"
data_edfig6d="${data_dir}/edfig6d_vison_model_rl_distance.csv"
data_edfig6e="${data_dir}/edfig6e_vison_model_rl_azimuth.csv"
data_edfig6f="${data_dir}/edfig6f_vison_model_rl_size.csv"

if [ ! -f $vid2 ] || [ ! -f $edfig2 ] || [ ! -f $data_ed2fig2 ]; then
    cd step_data
    jupyter nbconvert --to script stepping_illustration.ipynb
    python stepping_illustration.py
    rm stepping_illustration.py
    mv outputs/single_step.mp4 "../$vid2"
    mv outputs/single_step.pdf "../$edfig2"
    mv outputs/single_step.csv "../$data_ed2fig2"
    cd ..
fi

if [ ! -f $edfig3a ] || [ ! -f $edfig3b ] || [ ! -f $edfig3ci ] || [ ! -f $edfig3cii ] || \
[ ! -f $edfig3di ] || [ ! -f $edfig3dii ] || [ ! -f $edfig3ei ] || [ ! -f $edfig3eii ]; then
    cd visual_inputs
    jupyter nbconvert --to script calibration_environment.ipynb
    python calibration_environment.py
    rm calibration_environment.py
    mv outputs/calibration_env/birdeye_view.png "../$edfig3a"
    mv outputs/calibration_env/birdeye_zoom_view.png "../$edfig3b"
    mv outputs/calibration_env/raw_img_L.png "../$edfig3ci"
    mv outputs/calibration_env/raw_img_R.png "../$edfig3cii"
    mv outputs/calibration_env/corrected_img_L.png "../$edfig3di"
    mv outputs/calibration_env/corrected_img_R.png "../$edfig3dii"
    mv outputs/calibration_env/human_view_L.png "../$edfig3ei"
    mv outputs/calibration_env/human_view_R.png "../$edfig3eii"
    cd ..
fi

if [ ! -f $vid3 ]; then
    cd leg_adhesion
    jupyter nbconvert --to script force_visualization.ipynb
    python force_visualization.py
    rm force_visualization.py
    mv outputs/force_visualization.mp4 "../$vid3"
    cd ..
fi

if [ ! -f $vid4 ]; then
    cd leg_adhesion
    python generate_slope_datapts.py
    python merge_videos.py
    mv outputs/climbing.mp4 "../$vid4"
    cd ..
fi

if [ ! -f $vid5 ]; then
    cd control_signal
    jupyter nbconvert --to script cpg.ipynb
    python cpg.py
    rm cpg.py
    mv outputs/cpg.mp4 "../$vid5"
    cd ..
fi

if [ ! -f $vid6 ]; then
    cd control_signal
    jupyter nbconvert --to script rule_based.ipynb
    python rule_based.py
    rm rule_based.py
    mv outputs/rule_based.mp4 "../$vid6"
    cd ..
fi

if [ ! -f $vid7 ]; then
    cd control_signal
    jupyter nbconvert --to script hybrid.ipynb
    python hybrid.py
    rm hybrid.py
    mv outputs/hybrid.mp4 "../$vid7"
    cd ..
fi

if [ ! -f $vid8 ] || [ ! -f $fig2g ] || [ ! -f $data_fig2g ]; then
    cd controller_comparison
    python generate_datapts.py
    jupyter nbconvert --to script generate_figure.ipynb
    python generate_figure.py
    rm generate_figure.py
    python make_summary_video.py
    sh compress_video.sh
    rm outputs/controller_comparison.mp4
    mv outputs/controller_comparison_small.mp4 "../$vid8"
    mv outputs/speed_comparison.pdf "../$fig2g"
    mv outputs/speed_comparison.csv "../$data_fig2g"
    cd ..
fi

if [ ! -f $vid9 ] || [ ! -f $fig3c ]; then
    cd visual_inputs
    jupyter nbconvert --to script visual_taxis.ipynb
    python visual_taxis.py 0
    rm visual_taxis.py
    mv outputs/object_following_with_retina_images.mp4 "../$vid9"
    mv outputs/visual_taxis.pdf "../$fig3c"
    cd ..
fi

if [ ! -f $vid10 ] || [ ! -f $fig3d ]; then
    cd odor_inputs
    jupyter nbconvert --to script odor_taxis.ipynb
    python odor_taxis.py
    rm odor_taxis.py
    mv outputs/odor_taxis.mp4 "../$vid10"
    mv outputs/odor_taxis.pdf "../$fig3d"
    cd ..
fi

if [ ! -f $fig1b ] || [ ! -f $fig3bi ] || [ ! -f $fig3bii ]; then
    cd integrated_task
    jupyter nbconvert --to script env_demo.ipynb
    python env_demo.py
    rm env_demo.py
    mv outputs/env_overview.png "../$fig1b"
    mv outputs/vision_sim.pdf "../$fig3bi"
    mv outputs/behind_fly_view.png "../$fig3bii"
    cd ..
fi

if [ ! -f $fig2b ]; then
    cd leg_adhesion
    jupyter nbconvert --to script plot_climbing.ipynb
    python plot_climbing.py
    rm plot_climbing.py
    mv outputs/climbing.pdf "../$fig2b"
    cd ..
fi

if [ ! -f $fig2c ] || [ ! -f $data_fig2c ]; then
    cd leg_adhesion
    python generate_gainslope_dataset_multiprocessing.py
    jupyter nbconvert --to script critical_angle_plot.ipynb
    python critical_angle_plot.py
    rm critical_angle_plot.py
    cp outputs/critical_slope.pdf "../$fig2c"
    cp outputs/critical_slope.csv "../$data_fig2c"
    cd ..
fi

if [ ! -f $fig2d ]; then
    cd complex_terrain
    jupyter nbconvert --to script terrain_comparison.ipynb
    python terrain_comparison.py
    rm terrain_comparison.py
    mv outputs/terrains.pdf "../$fig2d"
    cd ..
fi

if [ ! -f $edfig6 ] || [ ! -f $data_edfig6c ] || [ ! -f $data_edfig6d ] || [ ! -f $data_edfig6e ] || [ ! -f $data_edfig6f ]; then
    cd integrated_task

    if [ ! -f "data/vision/visual_training_data.pkl" ]; then
        jupyter nbconvert --to script collect_visual_training_data.ipynb
        python collect_visual_training_data.py
        rm collect_visual_training_data.py
    fi

    jupyter nbconvert --to script train_vision_model.ipynb
    python train_vision_model.py
    rm train_vision_model.py

    mv outputs/vision_model_stats.pdf "../$edfig6"
    mv outputs/direction.csv "../$data_edfig6c"
    mv outputs/distance.csv "../$data_edfig6d"
    mv outputs/azimuth.csv "../$data_edfig6e"
    mv outputs/size.csv "../$data_edfig6f"

    cd ..
fi

if [ ! -f $vid13 ] || [ ! -f $fig5b ] || [ ! -f $fig5c ]; then
    cd integrated_task/preprint_trial

    if [ ! -f "data/rl_model.zip" ]; then
        python train_navigation_task.py
    fi

    python run_and_visualize.py
    python merge_videos.py

    jupyter nbconvert --to script generate_figures.ipynb
    python generate_figures.py
    rm generate_figures.py

    mv outputs/trajectory.pdf "../../$fig5b"
    mv outputs/trajectories.pdf "../../$fig5c"
    mv outputs/navigation_task_merged.mp4 "../../$vid13"
    cd ../..
fi
