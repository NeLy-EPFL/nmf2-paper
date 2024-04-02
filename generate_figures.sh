#!/bin/bash

video_dir="_outputs/VIDEOS"
figure_dir="_outputs/FIGURES"

svid1="${video_dir}/Video1_ForcesLocomotion/video_1_force_visualization_v6_TL.mp4"
svid2="${video_dir}/Video2_Climbing/video_2_climbing_v6_TL.mp4"
svid3="${video_dir}/Video3_SingleStep/video_3_single_step_v4_TL.mp4"
svid4="${video_dir}/Video4_CPG/video_4_cpg_controller_v7_TL.mp4"
svid5="${video_dir}/Video5_RuleBased/video_5_rule_based_controller_v6_TL.mp4"
svid8_no_stable="${video_dir}/Video8_VisualTaxis/video_8_visual_taxis_no_stable_v13_TL.mp4"
svid8_stable="${video_dir}/Video8_VisualTaxis/video_8_visual_taxis_stable_v13_TL.mp4"
svid9="${video_dir}/Video9_OdorTaxis/video_9_odor_taxis_v6_TL.mp4"

edfig2="${figure_dir}/EDFig2_PreprogrammedStepping/edfig2_preprogrammed_stepping_v6_TL.pdf"
fig3visual_no_stable="${figure_dir}/Fig3_VisionOlfactionRL/fig3_sensory_visual_taxis_no_stable_v12_TL.pdf"
fig3visual_stable="${figure_dir}/Fig3_VisionOlfactionRL/fig3_sensory_visual_taxis_stable_v12_TL.pdf"
fig3odor="${figure_dir}/Fig3_VisionOlfactionRL/fig3_sensory_odor_taxis_v12_TL.pdf"

mkdir -p "$(dirname $svid1)"
mkdir -p "$(dirname $svid2)"
mkdir -p "$(dirname $svid3)"
mkdir -p "$(dirname $svid4)"
mkdir -p "$(dirname $svid5)"
mkdir -p "$(dirname $svid8_no_stable)"
mkdir -p "$(dirname $svid8_stable)"
mkdir -p "$(dirname $svid9)"
mkdir -p "$(dirname $edfig2)"
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

# # supplementary video 2
# if [ ! -f $svid2 ]; then
#     python leg_adhesion/generate_slope_datapts.py
#     python leg_adhesion/merge_videos.py $svid2
# fi

# # supplementary video 3
# if [ ! -f $svid3 ] || [ ! -f $edfig2 ] ; then
#     cd ./step_data
#     jupyter nbconvert --to script stepping_illustration.ipynb
#     python stepping_illustration.py
#     rm stepping_illustration.py
#     cd ..
#     mv step_data/outputs/single_step.mp4 $svid3
#     mv step_data/outputs/single_step.pdf $edfig2
# fi

# # supplementary video 4
# if [ ! -f $svid4 ]; then
#     cd control_signal
#     jupyter nbconvert --to script CPG_control.ipynb
#     python CPG_control.py
#     rm CPG_control.py
#     cd ..
#     mv control_signal/outputs/cpg.mp4 $svid4
# fi

# # supplementary video 5
# if [ ! -f $svid5 ]; then
#     cd control_signal
#     jupyter nbconvert --to script decentralized_control.ipynb
#     python decentralized_control.py
#     rm decentralized_control.py
#     cd ..
#     mv control_signal/outputs/rule_based.mp4 $svid5
# fi

# # supplementary video 8
# if [ ! -f $svid8_no_stable ] || [ ! -f $fig3visual_no_stable ]; then
#     cd visual_inputs
#     python visual_taxis.py 0
#     cd ..
#     mv visual_inputs/outputs/object_following_with_retina_images.mp4 $svid8_no_stable
#     mv visual_inputs/outputs/visual_taxis.pdf $fig3visual_no_stable
# fi

# if [ ! -f $svid8_stable ] || [ ! -f $fig3visual_stable ]; then
#     cd visual_inputs
#     python visual_taxis.py 1000
#     cd ..
#     mv visual_inputs/outputs/object_following_with_retina_images.mp4 $svid8_stable
#     mv visual_inputs/outputs/visual_taxis.pdf $fig3visual_stable
# fi

# # supplementary video 9
# if [ ! -f $svid9 ] || [ ! -f $fig3odor ]; then
#     cd odor_inputs
#     jupyter nbconvert --to script odor_taxis.ipynb
#     python odor_taxis.py
#     rm odor_taxis.py
#     cd ..
#     mv odor_inputs/outputs/odor_taxis.mp4 $svid9
#     mv odor_inputs/outputs/odor_taxis.pdf $fig3odor
# fi
