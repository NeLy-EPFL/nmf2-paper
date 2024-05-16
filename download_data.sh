#!/bin/bash
visual_training_data_src=https://dataverse.harvard.edu/api/access/datafile/10163622
visual_preprocessor_src=https://dataverse.harvard.edu/api/access/datafile/10163626
rl_model_src=https://dataverse.harvard.edu/api/access/datafile/10163621

visual_training_data_dst=integrated_task/data/vision/visual_training_data.pkl
visual_preprocessor_dst=integrated_task/data/vision/visual_preprocessor.pt
rl_model_dst=integrated_task/preprint_trial/data/rl_model.zip

# Download visual training data
if [ ! -f $visual_training_data_dst ]; then
    mkdir -p $(dirname $visual_training_data_dst)
    wget -O $visual_training_data_dst $visual_training_data_src
fi

# Download visual preprocessor
if [ ! -f $visual_preprocessor_dst ]; then
    mkdir -p $(dirname $visual_preprocessor_dst)
    wget -O $visual_preprocessor_dst $visual_preprocessor_src
fi

# Download RL model
if [ ! -f $rl_model_dst ]; then
    mkdir -p $(dirname $rl_model_dst)
    wget -O $rl_model_dst $rl_model_src
fi