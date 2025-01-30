import numpy as np
import torch
import matplotlib.pyplot as plt


def energy(J, b, s):
    """
    Compute the energy of a given solution using the QUBO formulation.

    Args:
        J (torch.Tensor): Interaction matrix.
        b (torch.Tensor): Bias vector.
        s (torch.Tensor): Binary solution vector.

    Returns:
        torch.Tensor: Computed energy value.
    """
    return -0.5 * torch.einsum('in,ij,jn->n', s, J, s) - torch.einsum('in,ik->n', s, b)


def get_Jh(Qubo_matrix):
    """
    Convert the QUBO matrix into the J and h parameters for SimCIM.

    Args:
        matrix (numpy.ndarray): QUBO matrix.

    Returns:
        tuple: (J, h), where J is the interaction matrix and h is the bias vector.
    """
    Q = torch.tensor(Qubo_matrix, dtype=torch.float32)
    Q = 0.5 * (Q + Q.t())  # Ensure the QUBO matrix is symmetric
    Q_int = Q - torch.diag_embed(torch.diag(Q))  # Extract interaction terms

    J = -0.25 * Q_int
    h = -0.25 * Q_int.sum(1) - torch.diagonal(Q) * 0.5  # Extract bias

    h = h.reshape(-1, 1)

    return J, h


def get_value_simcim(x, matrix):
    """
    Compute the QUBO value for a given binary solution.

    Args:
        x (numpy.ndarray): Binary solution vector.
        matrix (numpy.ndarray): QUBO matrix.

    Returns:
        float: Computed energy value of the solution.
    """
    # Convert x to a binary vector if it contains -1 and 1
    x_binary = (0.5 * (x + 1)) if -1 in x else x

    # Calculate the QUBO energy value
    qubo_value = (np.dot(np.matmul(matrix + np.diag(np.diag(matrix)), x_binary), x_binary)) / 2

    return qubo_value  # Return the value as a scalar