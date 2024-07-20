import numpy as np
from tensornetwork import ncon
import scipy


def initialise_random_mps(N: int, d: int, chi: int):
    """
    Initialise an MPS with random real positive coefficients in the site tensors and identity weight matrices.

    Parameters:
    - N (int): Chain length
    - d (int): Local Hilbert space dimension (d = 2s + 1)
    - chi (int): Bond dimension

    Returns:
    - mps_tensors (list of numpy.ndarray): List of MPS site tensors. len(N) each of shape(chi, d, chi)
    - weights (list of numpy.ndarray): List of 1D-arrays representing the elements of the diagonal weight matrices on each bond. len(N) each of shape(chi)
    """
    # Generate random MPS
    rng = np.random.default_rng()
    mps_tensors = [rng.uniform(size=(chi, d, chi)) for _ in range(N)]
    weights = [np.ones((chi)) / np.sqrt(chi) for _ in range(N)]

    # Canonicalize MPS
    sigmas = left_environment_tensors(mps_tensors, weights)
    mus = right_environment_tensors(mps_tensors, weights)
    mps_tensors, weights = canonical_form(mps_tensors, weights, sigmas, mus)

    return mps_tensors, weights


def apply_two_body_gate(
    gate: np.ndarray,
    mps_tensor_left: np.ndarray,
    mps_tensor_right: np.ndarray,
    weight_left: np.ndarray,
    weight_center: np.ndarray,
    weight_right: np.ndarray,
    max_chi_trunc: np.ndarray,
    weight_tol: float = 1e-7,
):
    """
    Apply a two-body gate to two MPS sites and the weight between them.

    Parameters:
    - gate (numpy.ndarray): The gate to be applied. shape(d, d, d, d)
    - mps_tensor_left (numpy.ndarray): Left MPS site tensor to which the two-body gate is applied. shape(chi_1, d, chi_int)
    - mps_tensor_right (numpy.ndarray): Right MPS site tensor to which the two-body gate is applied. shape(chi_int, d, chi_2)
    - weight_left (numpy.ndarray): Diagonal matrix elements of weight on the left of mps_tensor_left. shape(chi_1)
    - weight_center (numpy.ndarray): Diagonal matrix elements of weight between mps_tensor_left and mps_tensor_right. shape(chi_int)
    - weight_right (numpy.ndarray): Diagonal matrix elements of weight on the right of mps_tensor_right. shape(chi_2)
    - max_chi_trunc (int): The maximum bond dimension to truncate to. chi_trunc=min(max_chi_trunc, min(d*chi_1, d*chi_2))
    - weight_tol (float): The minimum (tolerance) for single values in weight matrices to avoid division by zero.

    Returns:
    - new_mps_tensor_left (numpy.ndarray): New left MPS site tensor. shape(chi, d, chi_trunc)
    - new_weight_center (numpy.ndarray): New weights between new_mps_tensor_left and new_mps_tensor_right. shape(chi_trunc)
    - new_mps_tensor_right (numpy.ndarray): New right MPS site tensor. shape(chi_trunc, d, chi)
    """
    chi_1, d = mps_tensor_left.shape[:-1]
    chi_2 = mps_tensor_right.shape[-1]

    # Set single weight values below tolerance to tolerance to avoid division by zero later
    weight_left_toled = np.where(weight_left < weight_tol, weight_tol, weight_left)
    # weight_center_toled = np.where(weight_center < weight_tol, weight_tol, weight_center)
    weight_right_toled = np.where(weight_right < weight_tol, weight_tol, weight_right)

    # Contract MPS tensors with weights and gate into single (d*chi_1, d*chi_2) matrix
    tensors = [
        np.diag(weight_left_toled),
        mps_tensor_left,
        np.diag(weight_center),
        mps_tensor_right,
        np.diag(weight_right_toled),
        gate,
    ]
    connects = [[-1, 1], [1, 5, 2], [2, 4], [4, 6, 3], [3, -4], [-2, -3, 5, 6]]
    contracted = ncon(tensors, connects).reshape((d * chi_1, d * chi_2))

    # Apply singular value decomposition (SVD) to separate contracted matrix into product of three (d*chi, d*chi) square matrices u, s, and vh where s is diagonal.
    u, s, vh = np.linalg.svd(contracted, full_matrices=False)

    # Truncate contracted dimensions of u, s, and vh to desired accuracy chi_trunc
    chi_trunc = min(max_chi_trunc, s.shape[0])
    u_trunc = u[:, :chi_trunc].reshape(chi_1, d * chi_trunc)
    s_trunc = s[:chi_trunc]
    vh_trunc = vh[:chi_trunc, :].reshape(d * chi_trunc, chi_2)

    # Factorise u and vh to recover weight_left and weight_right
    new_mps_tensor_left = (np.diag(1 / weight_left_toled) @ u_trunc).reshape(
        chi_1, d, chi_trunc
    )
    new_mps_tensor_right = (vh_trunc @ np.diag(1 / weight_right_toled)).reshape(
        chi_trunc, d, chi_2
    )

    # Normalise new center weight
    new_weight_center = s_trunc / np.linalg.norm(s_trunc)

    return new_mps_tensor_left, new_weight_center, new_mps_tensor_right


def one_body_gate_expectation_values(
    gate: np.ndarray, mps_tensors: list[np.ndarray], weights: list[np.ndarray]
):
    """
    Compute the expectation values of a one-body gate at every site from an infinite MPS in canonical form (!).

    Parameters:
    - gate (numpy.ndarray): The one-body gate to be applied. shape(d, d)
    - mps_tensors (list of numpy.ndarray): List of MPS site tensors. len(N) each of shape(chi, d, chi)
    - weights (list of numpy.ndarray): List of 1D-arrays representing the elements of the diagonal weight matrices on each bond. len(N) each of shape(chi)

    Returns:
    - vals (numpy.ndarray): Expectation value at each site. shape(N)
    """
    N = len(mps_tensors)
    vals = np.zeros(N)
    for i in range(N):
        tensors = [
            np.diag(weights[i - 1] ** 2),
            mps_tensors[i],
            mps_tensors[i].conj(),
            np.diag(weights[i] ** 2),
            gate,
        ]
        connects = [[1, 2], [2, 5, 3], [1, 6, 4], [3, 4], [6, 5]]
        vals[i] = ncon(tensors, connects).real
    return vals


def left_environment_tensors(mps_tensors: list[np.ndarray], weights: list[np.ndarray]):
    """
    Compute left environment tensors for an infinite MPS.

    Parameters:
    - mps_tensors (list of numpy.ndarray): List of MPS site tensors. len(N) each of shape(chi, d, chi)
    - weights (list of numpy.ndarray): List of 1D-arrays representing the elements of the diagonal weight matrices on each bond. len(N) each of shape(chi)

    Returns:
    - sigmas: List of left environment tensors for each bond. len(N) each of shape(chi, chi)
    """
    N = len(mps_tensors)
    chi = mps_tensors[0].shape[0]
    sigmas = [
        0 for _ in range(N)
    ]  # Initialise list to correct length to allow assignment

    # Build tensor network to find left transfer operator
    lto_connects = []
    lto_tensors = []
    for i in range(N):
        lto_tensors.append(np.diag(weights[i - 1]))
        lto_connects.append([5 * i + 1, 5 * i + 3])
        lto_tensors.append(np.diag(weights[i - 1]))
        lto_connects.append([5 * i + 2, 5 * i + 4])
        lto_tensors.append(mps_tensors[i])
        lto_tensors.append(mps_tensors[i].conj())

        # For the last two tensors use negative connection indices to indicate external dimensions
        if i == (N - 1):
            lto_connects.append([5 * i + 3, 5 * i + 5, -1])
            lto_connects.append([5 * i + 4, 5 * i + 5, -2])
        else:
            lto_connects.append([5 * i + 3, 5 * i + 5, 5 * i + 6])
            lto_connects.append([5 * i + 4, 5 * i + 5, 5 * i + 7])

    # Cast left transfer operator as linear operator and find its eigenvector sigma_left (sigma_{N-1})
    def left_transfer_operator(sigma_left):
        return ncon(
            [sigma_left.reshape((chi, chi)), *lto_tensors], [[1, 2], *lto_connects]
        ).reshape((chi**2, 1))

    left_transfer_lo = scipy.sparse.linalg.LinearOperator(
        (chi**2, chi**2), matvec=(left_transfer_operator)
    )
    eigval, sigma_left = scipy.sparse.linalg.eigs(
        left_transfer_lo,
        k=1,
        which="LM",
        tol=1e-10,
        v0=(np.eye(chi) / chi).reshape(chi**2),
    )

    # Normalise sigma_left
    sigma_left = sigma_left.reshape((chi, chi))
    sigma_left = 0.5 * (sigma_left + np.conj(sigma_left.T))
    sigmas[-1] = sigma_left / np.trace(sigma_left)

    # Calculate remaining sigmas by contraction
    for i in range(N - 1):
        tensors = [
            sigmas[i - 1],
            np.diag(weights[i - 1]),
            np.diag(weights[i - 1]),
            mps_tensors[i],
            mps_tensors[i].conj(),
        ]
        connects = [[1, 2], [1, 3], [2, 4], [3, 5, -1], [4, 5, -2]]
        sigma = ncon(tensors, connects)
        sigmas[i] = sigma / np.trace(sigma)

    return sigmas


def right_environment_tensors(mps_tensors: list[np.ndarray], weights: list[np.ndarray]):
    """
    Compute right environment tensors for an infinte MPS.

    Parameters:
    - mps_tensors (list of numpy.ndarray): List of MPS site tensors. len(N) each of shape(chi, d, chi)
    - weights (list of numpy.ndarray): List of 1D-arrays representing the elements of the diagonal weight matrices on each bond. len(N) each of shape(chi)

    Returns:
    - mus: List of right environment tensors for each bond. len(N) each of shape(chi, chi)
    """
    N = len(mps_tensors)
    chi = mps_tensors[0].shape[0]
    mus = [0 for _ in range(N)]  # Initialise list to correct length to allow assignment

    # Build tensor network to find right transfer operator
    rto_connects = []
    rto_tensors = []
    for i in range(N):
        rto_tensors.append(np.diag(weights[N - 1 - i]))
        rto_connects.append([5 * i + 3, 5 * i + 1])
        rto_tensors.append(np.diag(weights[N - 1 - i]))
        rto_connects.append([5 * i + 4, 5 * i + 2])
        rto_tensors.append(mps_tensors[N - 1 - i])
        rto_tensors.append(mps_tensors[N - 1 - i].conj())

        # For the last two tensors use negative connection indices to indicate external dimensions
        if i == (N - 1):
            rto_connects.append([-1, 5 * i + 5, 5 * i + 3])
            rto_connects.append([-2, 5 * i + 5, 5 * i + 4])
        else:
            rto_connects.append([5 * i + 6, 5 * i + 5, 5 * i + 3])
            rto_connects.append([5 * i + 7, 5 * i + 5, 5 * i + 4])

    # Cast right transfer operator as linear operator and find its eigenvector sigma_left (sigma_{N-1})
    def right_transfer_operator(mu_right: np.ndarray):
        return ncon(
            [mu_right.reshape((chi, chi)), *rto_tensors], [[1, 2], *rto_connects]
        ).reshape((chi**2, 1))

    right_transfer_lo = scipy.sparse.linalg.LinearOperator(
        (chi**2, chi**2), matvec=(right_transfer_operator)
    )
    eigval, mu_right = scipy.sparse.linalg.eigs(
        right_transfer_lo,
        k=1,
        which="LM",
        tol=1e-10,
        v0=(np.eye(chi) / chi).reshape(chi**2),
    )

    # Normalise mu_right
    mu_right = mu_right.reshape((chi, chi))
    mu_right = 0.5 * (mu_right + np.conj(mu_right.T))
    mus[-1] = mu_right / np.trace(mu_right)

    # Calculate remaining sigmas by contraction
    for i in range(N - 1):
        tensors = [
            mus[N - 1 - i],
            np.diag(weights[N - 1 - i]),
            np.diag(weights[N - 1 - i]),
            mps_tensors[N - 1 - i],
            mps_tensors[N - 1 - i].conj(),
        ]
        connects = [[1, 2], [3, 1], [4, 2], [-1, 5, 3], [-2, 5, 4]]
        mu = ncon(tensors, connects)
        mus[N - 2 - i] = mu / np.trace(mu)

    return mus


def canonical_form(
    mps_tensors: list[np.ndarray],
    weights: list[np.ndarray],
    sigmas: list[np.ndarray],
    mus: list[np.ndarray],
    dtol: float = 1e-12,
):
    """
    Apply a gauge transformation to the MPS such that all left and right environment tensors become the identity matrix. This is the canonical form.

    Parameters:
    - mps_tensors (list of numpy.ndarray): List of MPS site tensors. len(N) each of shape(chi, d, chi)
    - weights (list of numpy.ndarray): List of 1D-arrays representing the elements of the diagonal weight matrices on each bond. len(N) each of shape(chi)
    - sigmas (list of numpy.ndarray): Left environment tensors. len(N) each of shape(chi, chi)
    - mus (list of numpy.ndarray): Right environment tensors. len(N) each of shape(chi, chi)
    - dtol (float): Truncation threshold for eigenvalues of environment tensors.

    Returns:
    - new_mps_tensors (list of numpy.ndarray): List of canonicalized MPS site tensors. len(N) each of shape(chi, d, chi)
    - new_weights (list of numpy.ndarray): List of canoncialized MPS weights. len(N) each of shape(chi)
    """
    N = len(mps_tensors)
    new_weights = [w.copy() for w in weights]
    new_mps_tensors = [t.copy() for t in mps_tensors]

    for i in range(-1, N - 1):
        # Diagonalize left environment
        dtemp, utemp = np.linalg.eigh(sigmas[i])
        nonzero_eigvals_mask = np.abs(dtemp) > dtol
        DL = dtemp[nonzero_eigvals_mask][::-1].astype(
            np.cfloat
        )  # Reorder eigenvalues in descending order and truncate so that all eigenvalues are greater than dtol
        UL = utemp[:, nonzero_eigvals_mask][:, ::-1].astype(
            np.cfloat
        )  # Apply same operation to eigenvectors

        # Diagonalize right environment
        dtemp, utemp = np.linalg.eigh(mus[i])
        nonzero_eigvals_mask = np.abs(dtemp) > dtol
        DR = dtemp[nonzero_eigvals_mask][::-1].astype(
            np.cfloat
        )  # Reorder eigenvalues in descending order and truncate so that all eigenvalues are greater than dtol
        UR = utemp[:, nonzero_eigvals_mask][:, ::-1].astype(
            np.cfloat
        )  # Apply same operation to eigenvectors

        # Calculate new weight
        # weighted_mat = (
        #     np.diag(np.sqrt(DL))
        #     @ UL.conj().T
        #     @ np.diag(weights[i])
        #     @ UR
        #     @ np.diag(np.sqrt(DR))
        # )
        # U, stemp, Vh = np.linalg.svd(weighted_mat, full_matrices=False)
        # new_weights[i] = stemp / np.linalg.norm(stemp)

        # # Calculate gauge change matrices and implement gauge change on adjacent tensors
        # X = UL @ np.diag(1 / np.sqrt(DL).conj()) @ U
        # Y = Vh @ np.diag(1 / np.sqrt(DR).conj()) @ UR.conj().T
        # new_mps_tensors[i] = ncon([new_mps_tensors[i], X], [[-1, -2, 1], [1, -3]])
        # new_mps_tensors[i + 1] = ncon([Y, new_mps_tensors[i + 1]], [[-1, 1], [1, -2, -3]])

        # Calculate new weight
        weighted_mat = (
            np.diag(np.sqrt(DL))
            @ UL.T
            @ np.diag(new_weights[i])
            @ UR
            @ np.diag(np.sqrt(DR))
        )
        UBA, stemp, VhBA = np.linalg.svd(weighted_mat, full_matrices=False)
        new_weights[i] = stemp / np.linalg.norm(stemp)

        # Calculate gauge change matrices and implement gauge change on adjacent tensors
        x = np.conj(UL) @ np.diag(1 / np.sqrt(DL)) @ UBA
        y = np.conj(UR) @ np.diag(1 / np.sqrt(DR)) @ VhBA.T
        new_mps_tensors[i + 1] = ncon(
            [y, new_mps_tensors[i + 1]], [[1, -1], [1, -2, -3]]
        )
        new_mps_tensors[i] = ncon([new_mps_tensors[i], x], [[-1, -2, 2], [2, -3]])

    # Normalise MPS tensors
    for i in range(N):
        tensors = [
            np.diag(new_weights[i - 1] ** 2),
            new_mps_tensors[i],
            new_mps_tensors[i].conj(),
            np.diag(new_weights[i] ** 2),
        ]
        connects = [[1, 3], [1, 4, 2], [3, 4, 5], [2, 5]]
        norm = np.sqrt(ncon(tensors, connects))
        new_mps_tensors[i] /= norm

    return new_mps_tensors, new_weights


def reduced_density_matrices(mps_tensors: list[np.ndarray], weights: list[np.ndarray]):
    """
    Compute reduced density matrices from an infinte MPS in canonical form(!).

    Parameters:
    - mps_tensors (list of numpy.ndarray): List of MPS site tensors. len(N) each of shape(chi, d, chi)
    - weights (list of numpy.ndarray): List of 1D-arrays representing the elements of the diagonal weight matrices on each bond. len(N) each of shape(chi)

    Returns:
    - rhos (list of numpy.ndarray): List of reduced density matrices across each bond. len(N) each of shape(d, d, d, d)
    """
    N = len(mps_tensors)
    rhos = [0 for _ in range(N)]

    for i in range(-1, N - 1):
        tensors = [
            np.diag(weights[i - 1] ** 2),
            mps_tensors[i],
            np.diag(weights[i]),
            mps_tensors[i + 1],
            np.diag(weights[i + 1] ** 2),
            mps_tensors[i + 1].conj().T,
            np.diag(weights[i]),
            mps_tensors[i].conj().T,
        ]
        connects = [
            [4, 3],
            [3, -3, 1],
            [1, 7],
            [7, -4, 5],
            [5, 6],
            [6, -2, 8],
            [8, 2],
            [2, -1, 4],
        ]
        rhos[i] = ncon(tensors, connects)

    return rhos
