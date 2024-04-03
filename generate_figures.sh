#!/bin/bash

video_dir="_outputs/VIDEOS"
figure_dir="_outputs/FIGURES"

svid1="${video_dir}/Video1_ForcesLocomotion/video_1_force_visualization_v6_TL.mp4"
svid2="${video_dir}/Video2_Climbing/video_2_climbing_v7_TL.mp4"
svid3="${video_dir}/Video3_SingleStep/video_3_single_step_v5_TL.mp4"
svid4="${video_dir}/Video4_CPG/video_4_cpg_controller_v8_TL.mp4"
svid5="${video_dir}/Video5_RuleBased/video_5_rule_based_controller_v7_TL.mp4"
svid6="${video_dir}/Video6_Hybrid/video_6_hybrid_controller_v6_TL.mp4"
svid7="${video_dir}/Video7_ControllerCompare/video_7_controller_comparison_v7_TL_small.mp4"

svid8_no_stable="${video_dir}/Video8_VisualTaxis/video_8_visual_taxis_no_stable_v14_TL.mp4"
svid8_stable="${video_dir}/Video8_VisualTaxis/video_8_visual_taxis_stable_v14_TL.mp4"
svid9="${video_dir}/Video9_OdorTaxis/video_9_odor_taxis_v7_TL.mp4"

edfig2="${figure_dir}/EDFig2_PreprogrammedStepping/edfig2_preprogrammed_stepping_v7_TL.pdf"
edfig4="${figure_dir}/EDFig4_VisionModelRL/edfig4_vison_model_rl_v3_TL.pdf"

fig2comparison="${figure_dir}/Fig2_AdhesionLocomotion/fig2_locomotion_v18_TL.pdf"

fig3visual_no_stable="${figure_dir}/Fig3_VisionOlfactionRL/fig3_sensory_visual_taxis_no_stable_v14_TL.pdf"
fig3visual_stable="${figure_dir}/Fig3_VisionOlfactionRL/fig3_sensory_visual_taxis_stable_v14_TL.pdf"
fig3odor="${figure_dir}/Fig3_VisionOlfactionRL/fig3_sensory_odor_taxis_v14_TL.pdf"

mkdir -p "$(dirname $svid1)"
mkdir -p "$(dirname $svid2)"
mkdir -p "$(dirname $svid3)"
mkdir -p "$(dirname $svid4)"
mkdir -p "$(dirname $svid5)"
mkdir -p "$(dirname $svid6)"
mkdir -p "$(dirname $svid7)"
mkdir -p "$(dirname $svid8_no_stable)"
mkdir -p "$(dirname $svid8_stable)"
mkdir -p "$(dirname $svid9)"
mkdir -p "$(dirname $edfig2)"
mkdir -p "$(dirname $edfig4)"
mkdir -p "$(dirname $fig2comparison)"
mkdir -p "$(dirname $fig3visual_no_stable)"
mkdir -p "$(dirname $fig3visual_stable)"
mkdir -p "$(dirname $fig3odor)"

# supplementary video 1
if [ ! -f $svid1 ]; then
    cd leg_adhesion
    jupyter nbconvert --to script force_visualization.ipynb
    python force_visualization.py
    rm force_visualization.py
    mv outputs/force_visualization.mp4 "../$svid1"
    cd ..
fi

# supplementary video 2
if [ ! -f $svid2 ]; then
    cd leg_adhesion
    python generate_slope_datapts.py
    python merge_videos.py
    mv data/slope_front/climbing.mp4 "../$svid2"
    cd ..
fi

# # supplementary video 3
if [ ! -f $svid3 ] || [ ! -f $edfig2 ] ; then
    cd step_data
    jupyter nbconvert --to script stepping_illustration.ipynb
    python stepping_illustration.py
    rm stepping_illustration.py
    mv outputs/single_step.mp4 "../$svid3"
    mv outputs/single_step.pdf "../$edfig2"
    cd ..
fi

# supplementary video 4
if [ ! -f $svid4 ]; then
    cd control_signal
    jupyter nbconvert --to script CPG_control.ipynb
    python CPG_control.py
    rm CPG_control.py
    mv outputs/cpg.mp4 "../$svid4"
    cd ..
fi

# supplementary video 5
if [ ! -f $svid5 ]; then
    cd control_signal
    jupyter nbconvert --to script decentralized_control.ipynb
    python decentralized_control.py
    rm decentralized_control.py
    mv outputs/rule_based.mp4 "../$svid5"
    cd ..
fi

# supplementary video 6
if [ ! -f $svid6 ]; then
    cd control_signal
    jupyter nbconvert --to script hybrid_signal.ipynb
    python hybrid_signal.py
    rm hybrid_signal.py
    mv outputs/hybrid.mp4 "../$svid6"
    cd ..
fi

# supplementary video 7
if [ ! -f $svid7 ] || [ ! -f $fig2comparison ]; then
    cd hybrid_controller
    # sh generate_all_data.sh
    jupyter nbconvert --to script generate_figure.ipynb
    python generate_figure.py
    rm generate_figure.py
    python make_summary_video.py
    mv outputs/controller_comparison.mp4 "../$svid7"
    mv outputs/speed_comparison.pdf "../$fig2comparison"
    cd ..
fi

# supplementary video 8
if [ ! -f $svid8_no_stable ] || [ ! -f $fig3visual_no_stable ]; then
    cd visual_inputs
    jupyter nbconvert --to script visual_taxis.ipynb
    python visual_taxis.py 0
    rm visual_taxis.py
    mv outputs/object_following_with_retina_images.mp4 "../$svid8_no_stable"
    mv outputs/visual_taxis.pdf "../$fig3visual_no_stable"
    cd ..
fi

if [ ! -f $svid8_stable ] || [ ! -f $fig3visual_stable ]; then
    cd visual_inputs
    jupyter nbconvert --to script visual_taxis.ipynb
    python visual_taxis.py 1000
    rm visual_taxis.py
    mv outputs/object_following_with_retina_images.mp4 "../$svid8_stable"
    mv outputs/visual_taxis.pdf "../$fig3visual_stable"
    cd ..
fi

# supplementary video 9
if [ ! -f $svid9 ] || [ ! -f $fig3odor ]; then
    cd odor_inputs
    jupyter nbconvert --to script odor_taxis.ipynb
    python odor_taxis.py
    rm odor_taxis.py
    mv outputs/odor_taxis.mp4 "../$svid9"
    mv outputs/odor_taxis.pdf "../$fig3odor"
    cd ..
fi

# extended data figure 4
if [ ! -f $edfig4 ]; then
    cd integrated_task
    jupyter nbconvert --to script collect_visual_training_data.ipynb
    python collect_visual_training_data.py
    rm collect_visual_training_data.py

    jupyter nbconvert --to script train_vision_model.ipynb
    python train_vision_model.py
    rm train_vision_model.py

    mv outputs/vision_model_stats.pdf "../$edfig4"
    cd ..
fi
