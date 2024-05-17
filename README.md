# NeuroMechFly v2 Manuscript Repository

This repository contains the scripts to generate the figures and videos for the NeuroMechFly v2 manuscript.

## Instructions
Clone this repository:
```sh
git clone git@github.com:NeLy-EPFL/nmf2-paper.git
cd nmf2-paper
git checkout dev-v1.0.0
```

Create the conda environment using the provided `environment.yml` file:
```sh
conda env create --file environment.yml
```

If you are working on a machine without a display (e.g., a server), you will need to switch the renderer to EGL:
```sh
conda activate nmf2-paper
conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
conda deactivate
```

Some tasks require time-consuming data collection and training. To save time, you can download the data and trained models:
```sh
./download_data.sh
```

Generate the figures and videos by running the following script:
```sh
conda activate nmf2-paper
./generate_figures.sh
```

Note that the results may vary slightly due to differences in system architecture.
