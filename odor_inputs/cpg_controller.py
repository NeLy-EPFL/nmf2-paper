import numpy as np
import pkg_resources
import pickle
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo


class CPG:
    """Central Pattern Generator.

    Attributes
    ----------
    phase : np.ndarray
        Current phase of each oscillator, of size (n_oscillators,).
    amplitude : np.ndarray
        Current amplitude of each oscillator, of size (n_oscillators,).
    frequencies : np.ndarray
        Target frequency of each oscillator, of size (n_oscillators,).
    phase_biases : np.ndarray
        Phase bias matrix describing the target phase relations between the oscillators.
        Dimensions (n_oscillators,n_oscillators), set to values generating a tripod gait.
    coupling_weights : np.ndarray
        Coupling weights matrix between oscillators enforcing phase relations.
        Dimensions (n_oscillators,n_oscillators).
    rates : np.ndarray
        Convergence rates for the amplitudes.
    targ_ampl : np.ndarray
        Target amplitude for each oscillator, of size (n_oscillators,).
        Default value of 1.0, modulated by input to the step function.

    Parameters
    ----------
    timestep : float
        Timestep duration for the integration.
    n_oscillators : int
        Number of individual oscillators, by default 6.
    """

    def __init__(self, timestep, n_oscillators: int = 6):
        self.n_oscillators = n_oscillators
        # Random initializaton of oscillator states
        self.phase = np.random.rand(n_oscillators)
        a = np.random.randint(1)
        self.amplitude = np.repeat(a, n_oscillators)  # np.random.rand(n_oscillators)
        self.timestep = timestep
        # CPG parameters
        self.frequencies = 40 * np.ones(n_oscillators)
        self.phase_biases = (
            2
            * np.pi
            * np.array(
                [
                    [0, 0.5, 0, 0.5, 0, 0.5],
                    [0.5, 0, 0.5, 0, 0.5, 0],
                    [0, 0.5, 0, 0.5, 0, 0.5],
                    [0.5, 0, 0.5, 0, 0.5, 0],
                    [0, 0.5, 0, 0.5, 0, 0.5],
                    [0.5, 0, 0.5, 0, 0.5, 0],
                ]
            )
        )
        self.coupling_weights = (np.abs(self.phase_biases) > 0).astype(float) * 5.0
        self.rates = 1000 * np.ones(n_oscillators)

    def step(self, amplitude_modulation=[0, 0]):
        self.targ_ampl = np.repeat(amplitude_modulation + np.array([1, 1]), 3)
        self.phase, self.amplitude = self.euler_int(
            self.phase, self.amplitude, self.targ_ampl, timestep=self.timestep
        )

    def euler_int(self, prev_phase, prev_ampl, targ_ampl, timestep):
        dphas, dampl = self.phase_oscillator(prev_phase, prev_ampl, targ_ampl)
        phase = (prev_phase + timestep * dphas) % (2 * np.pi)
        ampl = prev_ampl + timestep * dampl
        return phase, ampl

    def phase_oscillator(self, phases, amplitudes, targ_ampl):
        """Phase oscillator model used in Ijspeert et al. 2007"""
        # NxN matrix with the phases of the oscillators
        phase_matrix = np.tile(phases, (self.n_oscillators, 1))

        # NxN matrix with the amplitudes of the oscillators
        amp_matrix = np.tile(amplitudes, (self.n_oscillators, 1))

        freq_contribution = 2 * np.pi * self.frequencies

        #  scaling of the phase differences between oscillators by the amplitude of the oscillators and the coupling weights
        scaling = np.multiply(amp_matrix, self.coupling_weights)

        # phase matrix and transpose substraction are analogous to the phase differences between oscillators, those should be close to the phase biases
        phase_shifts_contribution = np.sin(
            phase_matrix - phase_matrix.T - self.phase_biases
        )

        # Here we compute the contribution of the phase biases to the derivative of the phases
        # we mulitply two NxN matrices and then sum over the columns (all j oscillators contributions) to get a vector of size N
        coupling_contribution = np.sum(
            np.multiply(scaling, phase_shifts_contribution), axis=1
        )

        # Here we compute the derivative of the phases given by the equations defined previously.
        # We are using for that matrix operations to speed up the computation
        dphases = freq_contribution + coupling_contribution
        dphases = np.clip(dphases, 0, None)

        damplitudes = np.multiply(self.rates, targ_ampl - amplitudes)

        return dphases, damplitudes

    def reset(self):
        self.phase = np.random.rand(self.n_oscillators)
        self.amplitude = np.random.rand(self.n_oscillators)


class NMFCPG(NeuroMechFlyMuJoCo):
    """Wrapper for the NeuroMechFlyMujoco class for Reinforcement Learning.

    Attributes
    ----------
    nmf : NeuroMechFlyMuJoCo
        Underlying NMF object.
    num_dofs : int
        Number of controlled DOFs of the nmf.
    action_space : gymnasium.spaces.Box
        Action space for RL.
    objective_space : gymnasium.spaces.Box
        Objective space for RL.
    timer : int
        Current timestep of the simulation.
    cpg : CPG
        Central Pattern Generator to generate leg movements.
    prev_action : [float,float]
        Memory of previous action for reward computation.
    step_data :
        Stepping data for every limb from recordings, to be used to convert CPG states
        into joint positions.

    Parameters
    ----------
    n_oscillators : int
        Number of individual oscillators for the CPG, by default 6.
    n_stabilisation_steps : int
        Number of simulation steps during which no joint command is given, for CPG
        stabilisation, by default 1000.
    objective : string
        Type of objective, determining the observation space and reward function.
        Can be "track_obj" (default), which adds vision to the observation space, or
        "max_y".
    steps_per_action : int
        Number of simulation steps to perform per computed action.
    """

    def __init__(
        self,
        n_oscillators: int = 6,
        n_stabilisation_steps: int = 5000,
        **kwargs,
    ):
        # The underlying normal NMF environment
        super().__init__(**kwargs)
        # Number of dofs of the observation space
        self.num_dofs = len(self.actuated_joints)
        # Action space - 2 values (alphaL and alphaR)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        
        # CPG initialization
        self.cpg = CPG(self.timestep, n_oscillators)
        self.n_stabilisation_steps = n_stabilisation_steps
        for _ in range(n_stabilisation_steps):
            self.cpg.step()

        # Processing of joint trajectories reference from stepping data
        self._load_preprogrammed_stepping()

    def reset(self):
        self.cpg.reset()
        for _ in range(self.n_stabilisation_steps):
            self.cpg.step()
        return super().reset()

    def step(self, action):
        # Scaling of the action to go from [-1,1] -> [-0.5,0.5]
        action = 0.5 * np.array(action)

        # Compute joint positions from NN output
        joints_action = self.compute_joints(action)

        # Step simulation and get observation after it
        return super().step(joints_action)

    def compute_joints(self, action):
        """Turn NN output into joint position."""
        self.cpg.step(action)
        indices = self._cpg_state_to_joint_state()
        joints_action = {
            "joints": self.step_data[self.joint_ids, 0]
            + (
                self.step_data[self.joint_ids, indices]
                - self.step_data[self.joint_ids, 0]
            )
            * self.cpg.amplitude[self.match_leg_to_joints]
        }

        return joints_action

    def _load_preprogrammed_stepping(self):
        legs = ["LF", "LM", "LH", "RF", "RM", "RH"]
        n_joints = len(self.actuated_joints)
        self.joint_ids = np.arange(n_joints).astype(int)
        self.match_leg_to_joints = np.array(
            [
                i
                for joint in self.actuated_joints
                for i, leg in enumerate(legs)
                if leg in joint
            ]
        )

        # Load recorded data
        data_path = Path(pkg_resources.resource_filename("flygym", "data"))
        with open(data_path / "behavior" / "single_steps.pkl", "rb") as f:
            data = pickle.load(f)

        # Treatment of the pre-recorded data
        step_duration = len(data["joint_LFCoxa"])
        self.interp_step_duration = int(
            step_duration * data["meta"]["timestep"] / self.timestep
        )
        step_data_block_base = np.zeros(
            (len(self.actuated_joints), self.interp_step_duration)
        )
        measure_t = np.arange(step_duration) * data["meta"]["timestep"]
        interp_t = np.arange(self.interp_step_duration) * self.timestep
        for i, joint in enumerate(self.actuated_joints):
            step_data_block_base[i, :] = np.interp(interp_t, measure_t, data[joint])

        self.step_data = step_data_block_base.copy()

        for side in ["L", "R"]:
            self.step_data[
                self.actuated_joints.index(f"joint_{side}MCoxa")
            ] += np.deg2rad(
                10
            )  # Protract the midlegs
            self.step_data[
                self.actuated_joints.index(f"joint_{side}HFemur")
            ] += np.deg2rad(
                -5
            )  # Retract the hindlegs
            self.step_data[
                self.actuated_joints.index(f"joint_{side}HTarsus1")
            ] -= np.deg2rad(
                15
            )  # Tarsus more parallel to the ground (flexed) (also helps with the hindleg retraction)
            self.step_data[
                self.actuated_joints.index(f"joint_{side}FFemur")
            ] += np.deg2rad(
                15
            )  # Protract the forelegs (slightly to conterbalance Tarsus flexion)
            self.step_data[
                self.actuated_joints.index(f"joint_{side}FTarsus1")
            ] -= np.deg2rad(
                15
            )  # Tarsus more parallel to the ground (flexed) (add some retraction of the forelegs)

    def _cpg_state_to_joint_state(self):
        """From phase define what is the corresponding timepoint in the joint dataset
        In the case of the oscillator, the period is 2pi and the step duration is the period of the step
        We have to match those two"""
        period = 2 * np.pi
        # match length of step to period phases should have a period of period mathc this perios to the one of the step
        t_indices = np.round(
            np.mod(
                self.cpg.phase * self.interp_step_duration / period,
                self.interp_step_duration - 1,
            )
        ).astype(int)
        t_indices = t_indices[self.match_leg_to_joints]
        return t_indices