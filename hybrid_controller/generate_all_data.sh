export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

for arena in gapped blocks mixed; do
    python generate_datapts.py --arena=$arena --adhesion --n_exp=10 --n_procs=1
    python generate_datapts.py --arena=$arena --n_exp=10 --n_procs=1
done