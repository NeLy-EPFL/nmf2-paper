import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import CubicSpline
import flygym.mujoco
import flygym.mujoco.preprogrammed
from tqdm import trange
from pathlib import Path

def calculate_ddt(theta, r, w, phi, nu, R, alpha):
    """Given the current state variables theta, r and network parameters
    w, phi, nu, R, alpha, calculate the time derivatives of theta and r."""
    intrinsic_term = 2 * np.pi * nu
    phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    coupling_term = (r * w * np.sin(phase_diff - phi)).sum(axis=1)
    dtheta_dt = intrinsic_term + coupling_term
    dr_dt = alpha * (R - r)
    return dtheta_dt, dr_dt

class CPGNetwork:
    def __init__(
        self,
        timestep,
        intrinsic_freqs,
        intrinsic_amps,
        coupling_weights,
        phase_biases,
        convergence_coefs,
        init_phases=None,
        init_magnitudes=None,
        seed=0,
    ) -> None:
        """Initialize a CPG network consisting of N oscillators.

        Parameters
        ----------
        timestep : float
            The timestep of the simulation.
        intrinsic_frequencies : np.ndarray
            The intrinsic frequencies of the oscillators, shape (N,).
        intrinsic_amplitudes : np.ndarray
            The intrinsic amplitude of the oscillators, shape (N,).
        coupling_weights : np.ndarray
            The coupling weights between the oscillators, shape (N, N).
        phase_biases : np.ndarray
            The phase biases between the oscillators, shape (N, N).
        convergence_coefs : np.ndarray
            Coefficients describing the rate of convergence to oscillator
            intrinsic amplitudes, shape (N,).
        init_phases : np.ndarray, optional
            Initial phases of the oscillators, shape (N,). The phases are
            randomly initialized if not provided.
        init_magnitudes : np.ndarray, optional
            Initial magnitudes of the oscillators, shape (N,). The
            magnitudes are randomly initialized if not provided.
        seed : int, optional
            The random seed to use for initializing the phases and
            magnitudes.
        """
        self.timestep = timestep
        self.num_cpgs = intrinsic_freqs.size
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.coupling_weights = coupling_weights
        self.phase_biases = phase_biases
        self.convergence_coefs = convergence_coefs
        self.random_state = np.random.RandomState(seed)

        self.reset(init_phases, init_magnitudes)

        # Check if the parameters have the right shape
        assert intrinsic_freqs.shape == (self.num_cpgs,)
        assert coupling_weights.shape == (self.num_cpgs, self.num_cpgs)
        assert phase_biases.shape == (self.num_cpgs, self.num_cpgs)
        assert convergence_coefs.shape == (self.num_cpgs,)
        assert self.curr_phases.shape == (self.num_cpgs,)
        assert self.curr_magnitudes.shape == (self.num_cpgs,)

    def step(self):
        """Integrate the ODEs using Euler's method."""
        dtheta_dt, dr_dt = calculate_ddt(
            theta=self.curr_phases,
            r=self.curr_magnitudes,
            w=self.coupling_weights,
            phi=self.phase_biases,
            nu=self.intrinsic_freqs,
            R=self.intrinsic_amps,
            alpha=self.convergence_coefs,
        )
        self.curr_phases += dtheta_dt * self.timestep
        self.curr_magnitudes += dr_dt * self.timestep

    def reset(self, init_phases=None, init_magnitudes=None):
        """Reset the phases and magnitudes of the oscillators."""
        if init_phases is None:
            self.curr_phases = self.random_state.random(self.num_cpgs) * 2 * np.pi
        else:
            self.curr_phases = init_phases

        if init_magnitudes is None:
            self.curr_magnitudes = (
                self.random_state.random(self.num_cpgs) * self.intrinsic_amps
            )
        else:
            self.curr_magnitudes = init_magnitudes

def load_data(data_path):
    with open(data_path, "rb") as f:
        single_steps_data = pickle.load(f)
    preprogrammed_steps_length = len(single_steps_data["joint_LFCoxa"])
    preprogrammed_steps_timestep = single_steps_data["meta"]["timestep"]
    print(
        f"Preprogrammed steps have a length of {preprogrammed_steps_length} steps "
        f"at dt={preprogrammed_steps_timestep}s."
    )

    # Check that the data is consistent
    for k, v in single_steps_data.items():
        if k.startswith("joint_"):
            assert len(v) == preprogrammed_steps_length
            #assert v[0] == v[-1]

    # Interpolate the data
    phase_grid = np.linspace(0, 2 * np.pi, preprogrammed_steps_length)
    psi_funcs = {}
    for leg in legs:
        joint_angles = np.array(
            [single_steps_data[f"joint_{leg}{dof}"] for dof in dofs_per_leg]
        )
        psi_funcs[leg] = CubicSpline(phase_grid, joint_angles, axis=1, bc_type="periodic")

    swing_start = np.empty(6)
    swing_end = np.empty(6)
    start_type = [[] for _ in range(6)]
    for i, leg in enumerate(legs):
        swing_start[i] = single_steps_data["swing_stance_time"]["swing"][leg]
        swing_end[i] = single_steps_data["swing_stance_time"]["stance"][leg]
        start_type[i] = single_steps_data["meta"]["starts"][leg[-1]]
    swing_start /= preprogrammed_steps_length * preprogrammed_steps_timestep
    swing_start *= 2 * np.pi
    swing_end /= preprogrammed_steps_length * preprogrammed_steps_timestep
    swing_end *= 2 * np.pi

    psi_rest_phases = np.ones_like(swing_start)
    for i, leg in enumerate(legs):
        psi_rest_phases[i] = (swing_end[i] + 2*np.pi) / 2
        
    return psi_funcs, swing_start, swing_end, psi_rest_phases

def get_adhesion_onoff(theta, swing_start, swing_end):
    theta = theta % (2 * np.pi)
    return ~((theta > swing_start) & (theta < swing_end)).squeeze()

###### CST

ADHESION = True

legs = [f"{side}{pos}" for side in "LR" for pos in "FMH"]
dofs_per_leg = [
    "Coxa",
    "Coxa_roll",
    "Coxa_yaw",
    "Femur",
    "Femur_roll",
    "Tibia",
    "Tarsus1",
]

intrinsic_freqs = np.ones(6) * 12
intrinsic_amps = np.ones(6) * 1
phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
coupling_weights = (phase_biases > 0) * 10
convergence_coefs = np.ones(6) * 5

run_time = 1

psi_base_phase = np.pi

def main():
    # Loading the data
    single_step_directory = Path("/Users/stimpfli/Desktop/nmf2-paper/revision_stepping/data/single_step_datasets")
    single_steps_paths = sorted(single_step_directory.glob("*.pkl"))

    cpg_network = CPGNetwork(
        timestep=1e-4,
        intrinsic_freqs=intrinsic_freqs,
        intrinsic_amps=intrinsic_amps,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=convergence_coefs,
    )

    sim_params = flygym.mujoco.Parameters(
        timestep=1e-4, render_mode="saved", render_playspeed=0.1, draw_adhesion=ADHESION, enable_adhesion=ADHESION, actuator_kp=50, tarsus_damping=10.0, tarsus_stiffness=10.0
    )

    for single_steps_path in single_steps_paths:

        psi_funcs, swing_start, swing_end, psi_rest_phases = load_data(single_steps_path)
        nmf = flygym.mujoco.NeuroMechFly(
            sim_params=sim_params,
            init_pose="tripod",
            actuated_joints=flygym.mujoco.preprogrammed.all_leg_dofs,
            control="position",
            xml = "mjcf_ikpy_model",
            
        )

        cpg_network.random_state = np.random.RandomState(seed=0)
        cpg_network.reset()
        bs, info = nmf.reset(seed=0)
        
        for _ in trange(int(run_time / sim_params.timestep)):
            cpg_network.step()
            joints_angles = {}
            for i, leg in enumerate(legs):
                psi = psi_funcs[leg](cpg_network.curr_phases[i])
                psi_base = psi_funcs[leg](psi_rest_phases[i])
                adjusted_psi = psi_base + (psi - psi_base) * cpg_network.curr_magnitudes[i]
                for dof, angle in zip(dofs_per_leg, adjusted_psi):
                    joints_angles[f"joint_{leg}{dof}"] = angle
            if ADHESION:
                adhesion_onoff = get_adhesion_onoff(cpg_network.curr_phases, swing_start, swing_end)
                action = {
                    "joints": np.array([joints_angles[dof] for dof in nmf.actuated_joints]),
                    ##### THIS LINE IS NEW #####
                    "adhesion": adhesion_onoff.astype(int),
                    ############################
                }
            else:
                action = {
                    "joints": np.array([joints_angles[dof] for dof in nmf.actuated_joints]),
                }
            obs, reward, terminated, truncated, info = nmf.step(action)
            
            nmf.render()

        dist_travelled = nmf.spawn_pos[0] - obs["fly"][0][0]
        filename = f"theorical_tripod_{single_steps_path.stem}_{dist_travelled:.2f}"
        filename += "adhesion.mp4" if sim_params.enable_adhesion else ".mp4"
        nmf.save_video(f"./videos/leg_combinations/{filename}", stabilization_time=0)

    return None

if __name__ == "__main__":
    print("Running main")
    main()