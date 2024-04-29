#!/bin/bash
video_dir="_outputs/VIDEOS"
figure_dir="_outputs/FIGURES"

mkdir -p $video_dir
mkdir -p $figure_dir

vid2="${video_dir}/video_2_single_step_v5_TL.mp4"
vid3="${video_dir}/video_3_force_visualization_v7_TL.mp4"
vid4="${video_dir}/video_4_climbing_v8_TL.mp4"
vid5="${video_dir}/video_5_cpg_controller_v9_TL.mp4"
vid6="${video_dir}/video_6_rule_based_controller_v8_TL.mp4"
vid7="${video_dir}/video_7_hybrid_controller_v10_TL.mp4"
vid8="${video_dir}/video_8_controller_comparison_v10_TL_small.mp4"
vid9="${video_dir}/video_9_visual_taxis_v14_TL.mp4"
vid10="${video_dir}/video_10_odor_taxis_v8_TL.mp4"
vid13="${video_dir}/video_13_multimodal_navigation_example_v3_TL.mp4"

edfig2="${figure_dir}/edfig2_preprogrammed_stepping_v8_TL.pdf"
edfig4="${figure_dir}/edfig4_vison_model_rl_v5_TL.pdf"

fig1b="${figure_dir}/fig1_schematics_env_overview_v18_TL.png"

fig2b="${figure_dir}/fig2_locomotion_climbing_v18_TL.pdf"
fig2d="${figure_dir}/fig2_locomotion_terrains_v18_TL.pdf"
fig2c="${figure_dir}/fig2_locomotion_critical_slope_v18_TL.pdf"
fig2g="${figure_dir}/fig2_locomotion_controller_comparison_v19_TL.pdf"

fig3bi="${figure_dir}/fig3_sensory_vision_sim_v15_TL.pdf"
fig3bii="${figure_dir}/fig3_sensory_behind_fly_view_v15_TL.png"
fig3c="${figure_dir}/fig3_sensory_visual_taxis_v14_TL.pdf"
fig3d="${figure_dir}/fig3_sensory_odor_taxis_v14_TL.pdf"

fig5b="${figure_dir}/fig5_integration_trajectory_v14_TL.pdf"
fig5c="${figure_dir}/fig5_integration_trajectories_v14_TL.pdf"

if [ ! -f $vid2 ] || [ ! -f $edfig2 ] ; then
    cd step_data
    jupyter nbconvert --to script stepping_illustration.ipynb
    python stepping_illustration.py
    rm stepping_illustration.py
    mv outputs/single_step.mp4 "../$vid2"
    cp outputs/single_step.pdf "../$edfig2"
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

if [ ! -f $vid8 ] || [ ! -f $fig2g ]; then
    cd hybrid_controller
    python generate_datapts.py
    jupyter nbconvert --to script generate_figure.ipynb
    python generate_figure.py
    rm generate_figure.py
    python make_summary_video.py
    sh compress_video.sh
    rm outputs/controller_comparison.mp4
    mv outputs/controller_comparison_small.mp4 "../$vid8"
    cp outputs/speed_comparison.pdf "../$fig2g"
    cd ..
fi

if [ ! -f $vid9 ] || [ ! -f $fig3c ]; then
    cd visual_inputs
    jupyter nbconvert --to script visual_taxis.ipynb
    python visual_taxis.py 0
    rm visual_taxis.py
    mv outputs/object_following_with_retina_images.mp4 "../$vid9"
    cp outputs/visual_taxis.pdf "../$fig3c"
    cd ..
fi

if [ ! -f $vid10 ] || [ ! -f $fig3d ]; then
    cd odor_inputs
    jupyter nbconvert --to script odor_taxis.ipynb
    python odor_taxis.py
    rm odor_taxis.py
    mv outputs/odor_taxis.mp4 "../$vid10"
    cp outputs/odor_taxis.pdf "../$fig3d"
    cd ..
fi

if [ ! -f $fig1b ] || [ ! -f $fig3bi ] || [ ! -f $fig3bii ]; then
    cd integrated_task
    jupyter nbconvert --to script env_demo.ipynb
    python env_demo.py
    rm env_demo.py
    cp outputs/env_overview.png "../$fig1b"
    cp outputs/vision_sim.pdf "../$fig3bi"
    cp outputs/behind_fly_view.png "../$fig3bii"
    cd ..
fi

if [ ! -f $fig2b ]; then
    cd leg_adhesion
    jupyter nbconvert --to script plot_climbing.ipynb
    python plot_climbing.py
    rm plot_climbing.py
    cp outputs/climbing.pdf "../$fig2b"
    cd ..
fi

if [ ! -f $fig2c ]; then
    cd leg_adhesion
    python generate_gainslope_dataset_multiprocessing.py
    jupyter nbconvert --to script critical_angle_plot.ipynb
    python critical_angle_plot.py
    rm critical_angle_plot.py
    cp outputs/critical_slope.pdf "../$fig2c"
    cd ..
fi

if [ ! -f $fig2d ]; then
    cd complex_terrain
    jupyter nbconvert --to script terrain_comparison.ipynb
    python terrain_comparison.py
    rm terrain_comparison.py
    cp outputs/terrains.pdf "../$fig2d"
    cd ..
fi

if [ ! -f $edfig4 ]; then
    cd integrated_task

    if [ ! -f "data/vision/visual_training_data.pkl" ]; then
        jupyter nbconvert --to script collect_visual_training_data.ipynb
        python collect_visual_training_data.py
        rm collect_visual_training_data.py
    fi

    jupyter nbconvert --to script train_vision_model.ipynb
    python train_vision_model.py
    rm train_vision_model.py

    cp outputs/vision_model_stats.pdf "../$edfig4"
    cd ..
fi

if [ ! -f $vid13 ] || [ ! -f $fig5b ] || [ ! -f $fig5c ]; then
    cd integrated_task/preprint_trial
    python train_navigation_task.py
    python run_and_visualize.py
    python merge_videos.py

    jupyter nbconvert --to script generate_figures.ipynb
    python generate_figures.py
    rm generate_figures.py

    cp outputs/trajectory.pdf "../../$fig5b"
    cp outputs/trajectories.pdf "../../$fig5c"
    mv outputs/navigation_task_merged.mp4 "../../$vid13"
    cd ../..
fi
