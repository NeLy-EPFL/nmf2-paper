from flygym import Fly, Camera, SingleFlySimulation
from flygym.examples.common import PreprogrammedSteps
from flygym.examples.cpg_controller import CPGNetwork
import numpy as np
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm


preprogrammed_steps = PreprogrammedSteps()

###### CONSTANTS ######
STABILIZATION_DUR = 0.2
GRAVITY_SWITCHING_T = 0.4

LEGS = ["RF", "RM", "RH", "LF", "LM", "LH"]
N_OSCILLATORS = len(LEGS)

COUPLING_STRENGTH = 10.0
AMP_RATES = 20.0
TARGET_AMPLITUDE = 1.0

RUN_TIME = 1


####### CPG #########
def get_CPG_parameters(freq=12):
    frequencies = np.ones(N_OSCILLATORS) * freq

    # For now each oscillator have the same amplitude
    target_amplitudes = np.ones(N_OSCILLATORS) * TARGET_AMPLITUDE
    rates = np.ones(N_OSCILLATORS) * AMP_RATES

    phase_biases = np.diff(np.mgrid[:6, :6], axis=0)[0] % 2 * np.pi
    coupling_weights = (np.abs(phase_biases) > 0).astype(float) * COUPLING_STRENGTH

    return frequencies, target_amplitudes, rates, phase_biases, coupling_weights


def run_cpg(
    adhesion_force: float,
    slope: float,
    seed: int,
    save_path: Path,
    debug=False,
):
    if save_path.exists():
        return

    fly = Fly(
        actuator_gain=45,
        enable_adhesion=True,
        draw_adhesion=True,
        adhesion_force=adhesion_force,
    )

    if debug:
        cam = Camera(
            fly=fly,
            play_speed=0.1,
            align_camera_with_gravity=True,
            camera_id="Animat/camera_left",
        )
        cameras = [cam]
    else:
        cameras = []

    sim = SingleFlySimulation(
        fly=fly,
        cameras=cameras,
        timestep=1e-4,
    )

    sim.reset()

    n_stabilization_steps = int(STABILIZATION_DUR / sim.timestep)
    gravity_switching_step = int(GRAVITY_SWITCHING_T / sim.timestep)
    num_steps = int(RUN_TIME / sim.timestep)

    # Get CPG parameters
    (
        frequencies,
        target_amplitudes,
        rates,
        phase_biases,
        coupling_weights,
    ) = get_CPG_parameters()

    # Initilize the simulation
    start_amps = np.ones(6) * 0.2

    cpg_network = CPGNetwork(
        timestep=sim.timestep,
        intrinsic_freqs=frequencies,
        intrinsic_amps=start_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=rates,
        init_magnitudes=start_amps,
        seed=seed,
    )

    # Initalize storage
    obs_fly_hist = []

    for i in range(num_steps):
        cpg_network.step()
        phase = cpg_network.curr_phases
        amp = cpg_network.curr_magnitudes

        if i == n_stabilization_steps:
            # Now set the amplitude to their real values
            cpg_network.intrinsic_amps[:] = target_amplitudes
        if i == gravity_switching_step:
            sim.set_slope(slope, "y")
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
            obs = sim.step(action)[0]
            obs_fly_hist.append(obs["fly"])
        except Exception as e:
            print(e)
            break

        if debug:
            sim.render()

    # Save the data
    np.savez_compressed(save_path, fly=np.array(obs_fly_hist, dtype=np.float32))

    if debug:
        cam.save_video(save_path.with_suffix(".mp4"), stabilization_time=0)


if __name__ == "__main__":
    forces = np.arange(0, 61, 5)
    slopes_in_degrees = np.arange(0, 181, 5)
    seeds = np.arange(5)

    output_path = Path("outputs/datapts_force_slope")
    output_path.mkdir(exist_ok=True, parents=True)

    it = product(forces, slopes_in_degrees, seeds)

    it = [
        (*i, output_path / "force_{:02d}_slope_{:03d}_seed_{}.npz".format(*i))
        for i in it
    ]

    Parallel(n_jobs=-1)(delayed(run_cpg)(*i, True) for i in tqdm(it))
