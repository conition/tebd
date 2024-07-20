import numpy as np
from tensornetwork import ncon
import mps


def generate_spin_operators(s: float):
    """
    Generates the spin operators (sx, sy, sz) for a given spin quantum number 's'.

    Parameters:
    - s (float): The spin quantum number.

    Returns:
    - tuple: A tuple containing the x, y, and z components of the spin operators. Each has shape (2*s+1, 2*s+1)
    """
    dim = 2 * s + 1  # Dimension of each spin operator

    a = np.arange(1, dim + 1, dtype=np.cfloat)[:, None]  # Row indices
    b = np.arange(1, dim + 1, dtype=np.cfloat)[None, :]  # Column indices

    sx = (
        0.5
        * ((a == (b + 1)) * 1 + ((a + 1) == b) * 1)
        * np.sqrt((s + 1) * (a + b - 1) - a * b)
    )
    sy = (
        0.5j
        * ((a == (b + 1)) * 1 - ((a + 1) == b) * 1)
        * np.sqrt((s + 1) * (a + b - 1) - a * b)
    )
    sz = (a == b) * (s + 1 - a)

    return sx, sy, sz


def do_tebd_timesteps(
    num_timesteps: int,
    gate: np.ndarray,
    mps_tensors: list[np.ndarray],
    weights: list[np.ndarray],
    max_chi_trunc: int,
):
    """
    Evolve state by num_timesteps time steps using TEBD with first-order Suzuki-Trotter expansion.

    Parameters:
    - num_timesteps (int): Number of timesteps to perform.
    - gate (numpy.ndarray): Two-body time evolution gate to be applied. shape(d, d, d, d)
    - mps_tensors (list of numpy.ndarray): List of MPS site tensors. len(N) each of shape(chi, d, chi)
    - weights (list of numpy.ndarray): List of 1D-arrays representing the elements of the diagonal weight matrices on each bond. len(N) each of shape(chi)
    - max_chi_trunc (int): The maximum bond dimension to truncate to.

    Returns:
    - new_mps_tensors (list of numpy.ndarray): List of updated MPS site tensors. len(N) each of shape(chi_trunc, d, chi_trunc)
    - new_weights (list of numpy.ndarray): List of updated MPS weights. len(N) each of shape(chi_trunc)
    """
    N = len(mps_tensors)
    new_mps_tensors = [t.copy() for t in mps_tensors]
    new_weights = [w.copy() for w in weights]
    for k in range(num_timesteps):
        for i in range(0, N, 2):
            new_mps_tensors[i], new_weights[i], new_mps_tensors[i + 1] = (
                mps.apply_two_body_gate(
                    gate,
                    new_mps_tensors[i],
                    new_mps_tensors[i + 1],
                    new_weights[i - 1],
                    new_weights[i],
                    new_weights[i + 1],
                    max_chi_trunc,
                    weight_tol=1e-7,
                )
            )

        for i in range(-1, N - 1, 2):
            new_mps_tensors[i], new_weights[i], new_mps_tensors[i + 1] = (
                mps.apply_two_body_gate(
                    gate,
                    new_mps_tensors[i],
                    new_mps_tensors[i + 1],
                    new_weights[i - 1],
                    new_weights[i],
                    new_weights[i + 1],
                    max_chi_trunc,
                    weight_tol=1e-7,
                )
            )
    return new_mps_tensors, new_weights


def do_tebd_midstep(
    mps_tensors: list[np.ndarray],
    weights: list[np.ndarray],
    nn_hamiltonian: np.ndarray,
    print_energy: bool = True,
):
    """
    Perform a TEBD midstep, canonicalizing the state for numerical stability and calculating and printing the energy.

    Parameters:
    - mps_tensors (list of numpy.ndarray): List of MPS site tensors. len(N) each of shape(chi, d, chi)
    - weights (list of numpy.ndarray): List of 1D-arrays representing the elements of the diagonal weight matrices on each bond. len(N) each of shape(chi)
    - nn_hamiltonian (np.ndarray): The nearest-neighbor Hamiltonian of the system, shape(d**2, d**2)
    - print_energy (bool): if True calculate and print total state energy

    Returns:
    - new_mps_tensors (list of np.ndarray): The updated list of MPS site tensors.
    - new_weights (list of np.ndarray): The updated list of MPS weights.
    """
    d = mps_tensors[0].shape[1]
    sigmas = mps.left_environment_tensors(mps_tensors, weights)
    mus = mps.right_environment_tensors(mps_tensors, weights)
    new_mps_tensors, new_weights = mps.canonical_form(mps_tensors, weights, sigmas, mus)
    if print_energy:
        rhos = mps.reduced_density_matrices(new_mps_tensors, new_weights)
        energy = ncon(
            [nn_hamiltonian.reshape((d, d, d, d)), np.array(rhos)],
            [[1, 2, 3, 4], [5, 1, 2, 3, 4]],
        )
        print(energy.real)
    return new_mps_tensors, new_weights
