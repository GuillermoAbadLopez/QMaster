# ruff: noqa: RUF002, RUF003, T201, CPY001

import itertools
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from extended_ising_model import ExtendedIsingModel, ExtendedIsingModelSweeper
from matplotlib import cm
from tenpy.algorithms import ExpMPOEvolution, TwoSiteDMRGEngine
from tenpy.models.spins_nnn import SpinChainNNN2
from tenpy.networks.mps import MPS

####################################################################################################
################## MAIN Model and ansatz class, with algorithm implemented in it ###################
####################################################################################################


class ExtendedIsingModel(SpinChainNNN2):
    def __init__(self, model_params: dict | None = None):
        """Initialize the extended Ising model.

        The Hamiltonian is given by
            H = -∑ₙ [ σᶻₙ σᶻₙ₊₁ + λ σᶻₙ + p (σᶻₙ σᶻₙ₊₂ + λ σˣₙ σˣₙ₊₁) ].

        Parameters
        ----------
            model_params (dict): A dictionary containing the model parameters. The following keys are recognized:
                L (int):  Number of Grouped sites (`site.GroupedSite`), at each unit cell. i.e. There will be 2*L total sites. Defaults to 2.
                S (float): The 2S+1 local states range from m = -S, -S+1, ... +S, can take values: {0.5, 1, 1.5, 2, ...}. Defaults to 0.5.
                p (float): The next-nearest neighbor strength for sigma_z and close neighbor sigma_x interactions; p=0 gives the transverse field Ising model. Defaults to 0.0.
                lmbd (float): The field strength of Z, affecting also the close neighbor sigma_x interactions. Defaults to 1.0.
                bc_MPS (str): The boundary conditions for the MPS simulation. Defaults to "infinite".
                conserve (str | None): What should be conserved:  'best' | 'Sz' | 'parity' | None. Defaults to None.
        """
        # Set default parameters (Transverse field Ising model):
        defaults = {
            "L": 2,
            "S": 0.5,
            "p": 0.0,
            "lmbd": 1.0,
            "bc_MPS": "infinite",
            "conserve": None,
        }
        # Update default parameters with any provided ones:
        if model_params is not None:
            defaults |= model_params

        # Store the parameters as attributes.
        self.L: int = defaults["L"]
        """ int: Number of sites (`site.GroupedSite`), at each unit cell. Defaults to 2. """

        self.S: float = defaults["S"]
        """ float: The 2S+1 local states range from m = -S, -S+1, ... +S, can take values: {0.5, 1, 1.5, 2, ...}. Defaults to 0.5."""

        self.p: float = defaults["p"]
        """ float: The next-nearest neighbor strength for sigma_z and close neighbor sigma_x interactions; p=0 gives the transverse field Ising model. Defaults to 0.0. """

        self.lmbd: float = defaults["lmbd"]
        """ float: The field strength of Z, affecting also the close neighbor sigma_x interactions. Defaults to 1.0. """

        self.bc_MPS: str = defaults["bc_MPS"]
        """ str: The boundary conditions for the MPS simulation. Defaults to "infinite". """

        self.conserve: str | None = defaults["conserve"]
        """What should be conserved:  'best' | 'Sz' | 'parity' | None. Defaults to None."""

        # Initialize the model via the parent constructor.
        super().__init__(self.spin_chain_params)

    @property
    def spin_chain_params(self) -> dict:
        """Translate the class parameters to the SpinChainNNN model.
        The factor 2, is because SpinChainNNN uses 0.5 * pauli_i, while we use pauli_i.

        Returns
        -------
            dict: The parameters for the SpinChainNNN model.
        """
        return {
            "L": self.L,  # Number of Grouped sites,i.e. There will be 2*L total sites.
            "S": self.S,
            "Jx": -self.p * self.lmbd * 4,  # Close neighbor sigma_x interaction.
            "Jy": 0,
            "Jz": -4,  # Close neighbor sigma_z interaction.
            "hx": self.lmbd * 2,  # Transverse (sigma_z) field strength.
            "hy": 0,
            "hz": 0,
            "Jxp": 0,
            "Jyp": 0,
            "Jzp": -self.p * 4,  # Next-nearest neighbor sigma_z interaction.
            "bc_MPS": self.bc_MPS,
            "conserve": self.conserve,
        }

    def get_plus_ansatz(self, init_state: str) -> MPS:
        """Get an |++...++> ansatz for the ground state of the model.

        Parameters
        ----------
            init_state (str): The initial state to use for the ansatz. Can be "++" or "up".

        Returns
        -------
            MPS: An MPS representing the ansatz for the ground state.
        """
        # Define the |++> state, for each GroupedSite (Site*2):
        if init_state == "plus":
            init_state = np.array([1, 1]) / np.sqrt(2)
        else:
            init_state = np.array([1, 0])

        # Create the MPS for mulitple sites:
        ansatz = MPS.from_product_state(
            self.lat.mps_sites(),  # Same lattice than the model.
            [init_state] * self.L,  # State plus repeated N_sites times; |++..(2L)..++>
            bc=self.bc_MPS,
        )

        # Normalize the state and put in canonical form:
        ansatz.canonical_form()

        return ansatz

    def run_dmrg(self, state: MPS, chi: int = 32, max_sweeps: int = 10) -> tuple[float, MPS]:
        """Runs the DMRG algorithm on a state.

        Parameters
        ----------
            state (MPS): The initial state to optimize.
            chi (int): The bond dimension of the MPS. Defaults to 32.
            max_sweeps (int): The maximum number of DMRG sweeps. Defaults to 10.

        Returns
        -------
            E (float): The energy of the resulting ground state MPS.
            state (MPS): The MPS representing the ground state after the simulation.
        """
        dmrg_params = {
            "mixer": None,  # setting this to True helps to escape local minima
            "max_E_err": 1.0e-10,
            # "chi_list": {0: chi},  # Use a constant bond dimension chi.
            # "energy_convergence": 1.0e-10,  # Stop if energy is below this value.
            "max_sweeps": max_sweeps,
            "trunc_params": {
                "chi_max": chi,
                # "svd_min": 1.0e-10,
            },
            "verbose": True,
            # "combine": True,
        }
        return TwoSiteDMRGEngine(state, self, dmrg_params).run()  # E, state

    def run_time_evolution(self, state: MPS, max_time: float, dt: float, chi_max: int) -> dict:
        """Runs time evolution of the state under the Hamiltonian.

        Parameters
        ----------
            state (MPS): The initial state to evolve.
            max_time (float): The final time to evolve to.
            dt (float): The time step to use.
            chi_max (int): The maximum bond dimension to use.

        Returns
        -------
            dict: The results as returned by prepare_results_for_save.
        """
        MPO_evol_params = {
            # "N_steps": 1,
            "dt": dt,
            "order": 2,
            "compression_method": "SVD",
            "max_E_err": 1.0e-10,
            "trunc_params": {
                "chi_max": chi_max,
                "svd_min": 1.0e-12,
            },
            "verbose": True,
        }
        eng = ExpMPOEvolution(state, self, MPO_evol_params)
        data = None

        while eng.evolved_time < max_time:
            eng.run()

            # Make canonical form and measure the state:
            eng.psi.canonical_form()
            data = self.measure(eng, data)

        return data

    @staticmethod
    def measure(eng: ExpMPOEvolution, data: dict | None) -> dict:
        """Measure the entropy, expectation values and bond dimensions of the state.

        Parameters
        ----------
            eng (ExpMPOEvolution): The engine used to evolve the state.
            data (dict): The data to append the measurements to. Defaults to None.

        Returns
        -------
            dict: The updated data dictionary.
        """
        if data is None:
            data = {
                "t": [],
                "entropy": [],
                "Sx": [],
                "Sz": [],
                "trunc_err": [],
                "max_bond_dim": [],
                "correlation_length": [],
                "psi": [],
                "spectrum": [],
                "gaps": [],
                "gap_ratios": [],
            }

        # Store time axis:
        data["t"].append(eng.evolved_time)

        # Compute the half-chain entanglement entropy (using the central bond).
        S_vals = eng.psi.entanglement_entropy()
        data["entropy"].append(S_vals[len(S_vals) // 2])

        # Record the truncation error:
        data["trunc_err"].append(eng.trunc_err.eps)

        # Record the maximum bond dimension across all bonds.
        data["max_bond_dim"].append(max(eng.psi.chi))
        data["correlation_length"].append(eng.psi.correlation_length())

        # Measure local magnetizations. For translational invariance we average all samples:
        exp_val_x = eng.psi.expectation_value("Sx")
        exp_val_z = eng.psi.expectation_value("Sz")
        data["Sx"].append(sum(exp_val_x) / len(exp_val_x))
        data["Sz"].append(sum(exp_val_z) / len(exp_val_z))

        # Compute the entanglement spectrum over time:
        spectrum = eng.psi.entanglement_spectrum()[0] / 2
        data["spectrum"].append(spectrum)

        # Compute the gaps and gap ratios over time:
        gaps = ExtendedIsingModel.compute_gaps(spectrum)
        data["gaps"].append(gaps)
        data["gap_ratios"].append(gaps / gaps[1])

        return data

    @staticmethod
    def compute_gaps(ent_spectrum):
        """Compute the gaps for the 10 lowest eigenvalues of the entanglement spectrum.

        Parameters
        ----------
            ent_spectrum (np.ndarray): The entanglement spectrum.

        Returns
        -------
            np.ndarray: The gaps for the 10 lowest eigenvalues.
        """
        # Sort the entanglement spectrum in ascending order
        sorted_spectrum = np.sort(ent_spectrum)
        # Compute the gaps for the 10 lowest eigenvalues
        return sorted_spectrum[:10] - sorted_spectrum[0]


####################################################################################################
######### Sweeper class to sweep and plot Algorithms for a range of D, p and lambda values #########
####################################################################################################


class ExtendedIsingModelSweeper:
    @staticmethod
    def sweep_dmrg(
        Ds: list[int], p_values: Iterable[float], lmbd_values: Iterable[float], init_state: str = "up"
    ) -> tuple[dict[tuple[int, float, float], float], ...]:
        """Run DMRG for a range of bond dimensions D, p and lambda values.

        Parameters
        ----------
            Ds (list[int]): List of bond dimensions to sweep.
            p_values (Iterable[float]): List of p values to sweep.
            lmbd_values (Iterable[float]): List of lambda values to sweep.
            init_state (str): The initial state to use. Defaults to "up".

        Returns
        -------
            results_E (dict[tuple[int, float, float], float]): Dictionary containing the energy for each p, lambda and D.
            results_xi (dict[tuple[int, float, float], float]): Dictionary containing the correlation length xi for each p, lambda and D.
            results_S (dict[tuple[int, float, float], float]): Dictionary containing the entropy S for each p, lambda and D.
        """
        # dictionary to store E, xi, and S for each p, lambda and D
        results_xi, results_E, results_S = {}, {}, {}

        for p, lmbd, D in itertools.product(p_values, lmbd_values, Ds):
            results_xi[p, lmbd, D] = {}
            results_E[p, lmbd, D] = {}
            results_S[p, lmbd, D] = {}

            model_params = {"L": 2, "S": 0.5, "p": p, "lmbd": lmbd, "bc_MPS": "infinite", "conserve": None}
            model = ExtendedIsingModel(model_params)

            # Get an initial ansatz for the model:
            state = model.get_plus_ansatz(init_state)

            # Run the iMPS simulation using e.g. iDMRG with bond dimension D
            E, state = model.run_dmrg(state=state, chi=D)
            results_E[p, lmbd, D] = E

            # Compute correlation length xi
            xi = state.correlation_length()
            results_xi[p, lmbd, D] = xi

            # Compute entropy
            S = ExtendedIsingModelSweeper.compute_halfchain_entropy(state)
            results_S[p, lmbd, D] = S

        return results_E, results_xi, results_S

    @staticmethod
    def compute_halfchain_entropy(state: MPS) -> float:
        """Compute the half-chain entropy of the state.

        Parameters
        ----------
            state (MPS): The state for which to compute the entropy.

        Returns
        -------
            float: The half-chain entropy of the state.
        """
        # state.entanglement_entropy() returns a list of S for each bond; take the central one
        S_vals = state.entanglement_entropy()
        return S_vals[len(S_vals) // 2]

    @staticmethod
    def get_results_vs_p_and_lambda(
        D: int, results: dict, p_values: Iterable[float], lmbd_values: Iterable[float]
    ) -> np.ndarray:
        """Create a 2D plot of the energy as a function of p and lambda.

        Parameters
        ----------
            D (int): The bond dimension to plot.
            results (dict): Dictionary containing the energy or correlations for each p, lambda and D.
            p_values (Iterable[float]): List of p values to sweep.
            lmbd_values (Iterable[float]): List of lambda values to sweep.

        Returns
        -------
            np.ndarray: The new results structure as a function of p and lambda.
        """
        if len(p_values) == 1:
            R = np.zeros(len(lmbd_values))
            for j, lmbd in enumerate(lmbd_values):
                R[j] = results[p_values[0], lmbd, D]
        else:
            R = np.zeros((len(p_values), len(lmbd_values)))
            for i, p in enumerate(p_values):
                for j, lmbd in enumerate(lmbd_values):
                    R[i, j] = results[p, lmbd, D]

        return R

    @staticmethod
    def get_results_vs_D(Ds: Iterable[int], results: dict, p: float, lmbd: float) -> np.ndarray:
        """Create a 1D plot of the results as a function of D.

        Parameters
        ----------
            Ds (Iterable[int]): The bond dimensions to plot.
            results (dict): Dictionary containing the energy or correlations for each p, lambda and D.
            p (float): p to plot.
            lmbd (float): Lambda to plot.

        Returns
        -------
            np.ndarray: The new results structure as a function of D.
        """
        R = np.zeros(len(Ds))
        for j, D in enumerate(Ds):
            R[j] = results[p, lmbd, D]
        return R

    @staticmethod
    def get_critical_lambdas_vs_D(
        Ds: Iterable[int], results: dict, p: float, lmbd_values: Iterable[float]
    ) -> np.ndarray:
        """Find the critical lambdas as a function of D.

        Parameters
        ----------
            Ds (Iterable[int]): The bond dimensions to plot.
            results (dict): Dictionary containing the energy or correlations for each p, lambda and D.
            p (float): p to plot.
            lmbd_values (Iterable[float]): Lambdas to search for to plot.

        Returns
        -------
            np.ndarray: The new critical lambdas as a function of D.
        """
        lambdas = np.zeros((len(Ds), len(lmbd_values)))
        lambda_critical = np.zeros(len(Ds))
        for j, D in enumerate(Ds):
            for i, lmbd in enumerate(lmbd_values):
                lambdas[j, i] = results[p, lmbd, D]
            lambda_critical[j] = lmbd_values[np.argmax(lambdas[j])]
        return lambda_critical

    @staticmethod
    def plot_sweep_vs_p_and_lambda(
        results: dict,
        title: str,
        zlabel: str,
        Ds: list[int],
        p_values: Iterable[float],
        lmbd_values: Iterable[float],
        color: str = "plasma",
    ) -> None:
        """Plot the results of the sweep as a function of p and lambda.

        Parameters
        ----------
            results (dict): Dictionary containing the energy or correlations for each p, lambda and D.
            title (str): The title of the plot.
            zlabel (str): The label of the z-axis.
            Ds (list[int]): List of bond dimensions to sweep.
            p_values (Iterable[float]): List of p values to sweep.
            lmbd_values (Iterable[float]): List of lambda values to sweep.
        """
        # Post-process results:
        figure = plt.figure(figsize=(16, 12))
        ax = [figure.add_subplot(max(len(Ds) // 2, 1), 3, i, projection="3d") for i in range(1, len(Ds) + 1)]

        for i, D in enumerate(Ds):
            # Values to plot
            R = ExtendedIsingModelSweeper.get_results_vs_p_and_lambda(D, results, p_values, lmbd_values)
            L, P = np.meshgrid(lmbd_values, p_values)

            ax[i].plot_surface(L, P, R, cmap=color)
            ax[i].set_title(f"{title} vs lambda and p, for D = {D!s}")
            ax[i].set_xlabel("lambda")
            ax[i].set_ylabel("p")
            ax[i].set_zlabel(zlabel)
            # ax[i].view_init(15, -60)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cut_at_fixed_p_vs_lambda(
        results: dict,
        title: str,
        ylabel: str,
        Ds: list[int],
        p: float,
        lmbd_values: Iterable[float],
        init_state: str,
        color: str = "plasma",
    ) -> None:
        """Plot the results of the sweep as a function of p and lambda.

        Parameters
        ----------
            results (dict): Dictionary containing the energy or correlations for each p, lambda and D.
            title (str): The title of the plot.
            ylabel (str): The label of the y-axis.
            Ds (list[int]): List of bond dimensions to sweep.
            p (float): Fixed p value to plot.
            lmbd_values (Iterable[float]): List of lambda values to sweep.
        """
        # Post-process results:
        colors = cm._colormaps[color](np.linspace(0, 0.85, len(Ds)))

        for i, D in enumerate(Ds):
            # Values to plot
            R = ExtendedIsingModelSweeper.get_results_vs_p_and_lambda(D, results, [p], lmbd_values)

            plt.plot(lmbd_values, R, label=f"D = {D!s}", marker="o", linestyle="-", color=colors[i])
            plt.title(f"{title} vs lambda, for p = {p} and init_state = {init_state}")
            plt.xlabel("lambda")
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_vs_D_at_fixed_p_and_lambdas(
        results: dict,
        title: str,
        ylabel: str,
        Ds: list[int],
        p: float,
        lmbd_values: Iterable[float],
        init_state: str,
        color: str = "plasma",
    ) -> None:
        """Plot the results of the sweep as a function of D, for fixed p and lambda values.

        Parameters
        ----------
            results (dict): Dictionary containing the energy or correlations for each p, lambda and D.
            title (str): The title of the plot.
            ylabel (str): The label of the y-axis.
            Ds (list[int]): List of bond dimensions to sweep.
            p (float): Fixed p value to plot.
            lmbd_values (Iterable[float]): List of lambda values to sweep.
        """
        # Post-process results:
        colors = cm._colormaps[color](np.linspace(0, 0.85, len(lmbd_values)))

        for i, lmbd in enumerate(lmbd_values):
            # Values to plot
            R = ExtendedIsingModelSweeper.get_results_vs_D(Ds, results, p, lmbd)

            plt.plot(Ds, R, label=f"lambda = {lmbd:.5f}", marker="o", linestyle="-", color=colors[i])
            plt.title(f"{title}, for p = {p} and init_state = {init_state}")
            plt.xlabel("D")
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_critical_lambda_vs_D_at_fixed_p(
        results: dict,
        title: str,
        ylabel: str,
        Ds: list[int],
        p: float,
        lmbd_values: Iterable[float],
        init_state: str,
        color: str = "plasma",
    ) -> None:
        """Plot the critical lambas of the sweep as a function of D.

        Parameters
        ----------
            results (dict): Dictionary containing the energy or correlations for each p, lambda and D.
            title (str): The title of the plot.
            ylabel (str): The label of the y-axis.
            Ds (list[int]): List of bond dimensions to sweep.
            p (float): Fixed p value to plot.
            lmbd_values (Iterable[float]): List of lambda values to sweep.
        """
        # Values to plot
        critical_lambas = ExtendedIsingModelSweeper.get_critical_lambdas_vs_D(Ds, results, p, lmbd_values)

        if critical_lambas[0] < critical_lambas[1]:
            Ds = Ds[1:]
            critical_lambas = critical_lambas[1:]
        plt.plot(Ds, critical_lambas, marker="o", linestyle="-", color=color)
        plt.title(f"{title}, for p = {p} and init_state = {init_state}")
        plt.xlabel("D")
        plt.ylabel(ylabel)
        plt.grid(True)

        plt.tight_layout()
        plt.show()


####################################################################################################
################ Plotter class to plot the final time evolutions of the algorithms #################
####################################################################################################


class ExtIsingModelTimeEvolPlotter:
    @staticmethod
    def plot_time_evolution(results: dict, color: str) -> None:
        """Plot the results of the time evolution.

        Parameters
        ----------
            results (dict): Dictionary containing the time evolution data.
            color (str): The color palette to use.
        """
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        axs = [ax1, ax2, ax3, ax4]

        # Use plasma color palette
        colors = cm._colormaps[color](np.linspace(0.0, 0.95, 7))

        # All the data to plot:
        plot_dict = {
            "Magnetization": [
                ([2 * sx for sx in results["Sz"]], r"$\langle \sigma^x \rangle$"),
                ([2 * sz for sz in results["Sx"]], r"$\langle \sigma^z \rangle$"),
            ],
            "Correlation & Entropy": [
                ([np.mean(ent) for ent in results["entropy"]], "entropy"),
                (results["correlation_length"], "correlation length"),
            ],
            "Bond Dimension": [results["max_bond_dim"]],
            "Truncation Error": [results["trunc_err"]],
        }

        for i, (ylabel, datas) in enumerate(plot_dict.items()):
            if len(datas) == 1:
                axs[i].plot(results["t"], datas[0], marker="o", linestyle="-", color=colors[2])
            else:
                for j, data in enumerate(datas):
                    axs[i].plot(
                        results["t"], data[0], label=data[1], marker="o", linestyle="-", color=colors[2 + 2 * j]
                    )
                    axs[i].legend()
            axs[i].set_title(f"{ylabel} vs Time")
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel(ylabel)
            axs[i].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_gaps_over_time(results: dict[str, list], colors: str) -> list:
        """Plot the gaps and their ratios as functions of time.

        Parameters
        ----------
            results (dict[str, list]): Dictionary containing the time evolution data.
            colors (str): The color palette to use.

        Returns
        -------
            list: The entanglement spectra.
        """
        # Convert lists to numpy arrays for easier plotting
        gaps_over_time = [[gaps[i] for gaps in results["gaps"]] for i in range(min(10, len(results["gaps"][0])))]
        gap_ratios_over_time = [
            [gap_ratios[i] for gap_ratios in results["gap_ratios"]]
            for i in range(min(10, len(results["gap_ratios"][0])))
        ]

        # Plot the gaps and their ratios as functions of time
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Use plasma color palette
        colors = cm._colormaps[colors](np.linspace(0, 0.85, 10))

        # Plot gaps
        for i in range(min(10, len(results["gaps"][0]))):
            ax1.plot(results["t"], gaps_over_time[i], label=f"Δ{i}", color=colors[i])
        ax1.set_title("Gaps Δi over Time")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Gap Δi")
        ax1.legend()
        ax1.grid(True)

        # Plot gap ratios
        for i in range(min(10, len(results["gap_ratios"][0]))):
            ax2.plot(results["t"], gap_ratios_over_time[i], label=f"Δ{i}/Δ1", color=colors[i])
        ax2.set_title("Gap Ratios Δi/Δ1 over Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Gap Ratio Δi/Δ1")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_time_evolution_of_spectra(results: dict[str, list], color: str):
        """Plot the evolution of the entanglement spectrum over time.

        Parameters
        ----------
            results (dict[str, list]): Dictionary containing the time evolution data.
            color (str): The color palette to use.
        """
        # Convert the list of entanglement spectra into a NumPy array for easier slicing
        # Take only the first 10 eigenvalues from each spectrum
        ent_spec_array = [
            [spectrum[i] for spectrum in results["spectrum"]] for i in range(min(10, len(results["spectrum"][0])))
        ]
        # Create a figure with a larger size for better visibility
        plt.figure(figsize=(12, 7))
        colors = cm._colormaps[color](np.linspace(0, 0.85, 10))

        # Plot the first 10 entanglement spectrum values over time
        for i in range(min(10, len(results["spectrum"][0]))):
            plt.plot(
                results["t"],
                ent_spec_array[i],
                label=f"$e_{{{i + 1}}}$",
                color=colors[i],
                marker=".",
                markersize=3,
            )

        # Customize the plot
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Entanglement Spectrum $e_i = -\\log(\\lambda_i)$", fontsize=12)
        plt.title("Evolution of Entanglement Spectrum", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.show()
