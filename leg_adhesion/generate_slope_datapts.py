import pickle
import time
import yaml
from pathlib import Path

import numpy as np

from flygym import Camera, Fly, SingleFlySimulation
from flygym.examples.common import PreprogrammedSteps
from flygym.examples.cpg_controller import CPGNetwork


preprogrammed_steps = PreprogrammedSteps()

###### CONSTANTS ######
CONTROLLER_SEED = 42
STABILIZATION_DUR = 0.2
GRAVITY_SWITCHING_T = 0.4

LEGS = ["RF", "RM", "RH", "LF", "LM", "LH"]
N_OSCILLATORS = len(LEGS)

COUPLING_STRENGTH = 10.0
AMP_RATES = 20.0
TARGET_AMPLITUDE = 1.0

RUN_TIME = 0.5 + 0.2


####### CPG #########
def get_CPG_parameters(freq=12):
    frequencies = np.ones(N_OSCILLATORS) * freq

    # For now each oscillator have the same amplitude
    target_amplitudes = np.ones(N_OSCILLATORS) * TARGET_AMPLITUDE
    rates = np.ones(N_OSCILLATORS) * AMP_RATES

    phase_biases = np.diff(np.mgrid[:6, :6], axis=0)[0] % 2 * np.pi
    coupling_weights = (np.abs(phase_biases) > 0).astype(float) * COUPLING_STRENGTH

    return frequencies, target_amplitudes, rates, phase_biases, coupling_weights


def run_CPG(
    slope,
    axis,
    adhesion,
    base_path,
):
    print(f"Running CPG gravity {slope} {axis} adhesion {adhesion}")

    fly = Fly(
        actuator_kp=45,
        enable_adhesion=adhesion,
        draw_adhesion=adhesion,
    )

    cam = Camera(
        fly=fly,
        play_speed=0.1,
        align_camera_with_gravity=True,
        camera_id="Animat/camera_front" if axis == "x" else "Animat/camera_left",
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=1e-4,
    )

    # Define save path
    save_path = base_path / f"CPG_gravity_{slope}_{axis}_adhesion{adhesion}.pkl"
    if save_path.exists():
        print(f"CPG gravity {slope} {axis} already exists")
        return
    video_path = save_path.with_suffix(".mp4")

    sim.reset()

    n_stabilization_steps = int(STABILIZATION_DUR / sim.timestep)
    gravity_switching_step = int(GRAVITY_SWITCHING_T / sim.timestep)

    num_steps = int(RUN_TIME / sim.timestep) + n_stabilization_steps

    # Get CPG parameters
    (
        frequencies,
        target_amplitudes,
        rates,
        phase_biases,
        coupling_weights,
    ) = get_CPG_parameters()

    # Initilize the simulation
    np.random.seed(CONTROLLER_SEED)
    start_ampl = np.ones(6) * 0.2

    cpg_network = CPGNetwork(
        timestep=sim.timestep,
        intrinsic_freqs=frequencies,
        intrinsic_amps=start_ampl,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=rates,
        init_magnitudes=start_ampl,
        seed=CONTROLLER_SEED,
    )

    # Initalize storage
    obs_list = []

    for i in range(num_steps):
        cpg_network.step()
        phase = cpg_network.curr_phases
        amp = cpg_network.curr_magnitudes

        if i == n_stabilization_steps:
            # Now set the amplitude to their real values
            cpg_network.intrinsic_amps[:] = target_amplitudes
        if i == gravity_switching_step:
            sim.set_slope(slope, axis)
        if i <= n_stabilization_steps:
            phase = phase * 0

        joints_angles = []
        adhesion_onoff = []

        for i, leg in enumerate(preprogrammed_steps.legs):
            my_joints_angles = preprogrammed_steps.get_joint_angles(
                leg, phase[i], amp[i]
            )
            joints_angles.append(my_joints_angles)
            my_adhesion_onoff = preprogrammed_steps.get_adhesion_onoff(leg, phase[i])
            adhesion_onoff.append(my_adhesion_onoff)

        action = {
            "joints": np.concatenate(joints_angles),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }

        try:
            obs, _, _, _, _ = sim.step(action)
            obs_list.append(obs)
            _ = sim.render()
        except Exception as e:
            print(e)
            break
    if video_path:
        cam.save_video(video_path, stabilization_time=STABILIZATION_DUR - 0.05)

    # Save the data
    with open(save_path, "wb") as f:
        pickle.dump(obs_list, f)


########### MAIN ############
if __name__ == "__main__":
    from itertools import product
    from joblib import Parallel, delayed

    slopes_in_degrees = np.arange(0, 190, 10)[::-1]
    axis = "y"
    timestep = 1e-4

    metadata = {
        "controller_seed": CONTROLLER_SEED,
        "run_time": RUN_TIME,
        "stabilization_dur": STABILIZATION_DUR,
        "gravity_switching_t": GRAVITY_SWITCHING_T,
        "coupling_strength": COUPLING_STRENGTH,
        "amp_rates": AMP_RATES,
        "target_amplitude": TARGET_AMPLITUDE,
        "legs": LEGS,
        "n_oscillators": N_OSCILLATORS,
        "timestep": timestep,
    }

    # Create folder to save data points
    base_path = Path(f"data/slope_front")
    base_path.mkdir(parents=True, exist_ok=True)

    # save metadata
    metadata_path = base_path / "metadata.yml"
    if not metadata_path.exists():
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)

    start_exps = time.time()
    print("Starting front slope experiments")

    Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(run_CPG)(slope, axis, adhesion, base_path)
        for slope, adhesion in product(slopes_in_degrees, (True, False))
    )

    print(
        f"{len(slopes_in_degrees)} experiments took "
        f"{time.time()-start_exps:.2f} seconds"
    )
