import numpy as np
import scipy.sparse as sparse

from smooth_utils import nicp
from smooth_utils import arap


def solve_Y21_nicp(mesh1, mesh2, p2p_21, params):
    if p2p_21.ndim == 1:
        Y21 = mesh1.vertlist[p2p_21]
    else:
        Y21 = p2p_21 @ mesh1.vertlist

    _, Y21 = nicp.solve_nicp(mesh2, mesh1, p2p_21,
                             gamma=1, alpha=params['smooth_weight']/params['couple_weight'],
                             landmarks=None, lmks_weight=10,
                             nit=1, n_jobs=-1,
                             use_cotan=True, verbose=False)
    return Y21


def solve_Y21_exact(mesh1, mesh2, p2p_21, params):
    # print(mesh1.path, mesh2.path, p2p_21)
    if p2p_21.ndim == 1:
        Y21 = mesh1.vertlist[p2p_21]
    else:
        Y21 = p2p_21 @ mesh1.vertlist  # (n2, 3)

    # print(params.keys(), f'solver{solver_ind2}', hasattr(params, f'solver{solver_ind2}'))
    if 'solver_ind' in params.keys():

        B_mat = (params['couple_weight'] / (params['smooth_weight'] * mesh2.area)) * mesh2.A @ Y21

        Y21_new = params[f'solver{params["solver_ind"]}'](B_mat)

        return Y21_new

    raise ValueError("Not in the official code yet")


def solve_Y21_exact_bij(mesh1, mesh2, p2p_21, p2p_12, params, pb_or_def='pull-back'):
    w_couple = params['couple_weight']
    w_bij = params['bij_weight']
    w_smooth = params['smooth_weight']

    if p2p_12.ndim == 1:
        P12 = sparse.csr_matrix((np.ones(p2p_12.size),(np.arange(p2p_12.size), p2p_12)), shape=(mesh1.n_vertices, mesh2.n_vertices))
    else:
        P12 = p2p_12

    if p2p_21.ndim == 1:
        P21 = sparse.csr_matrix((np.ones(p2p_21.size),(np.arange(p2p_21.size), p2p_21)), shape=(mesh2.n_vertices, mesh1.n_vertices))
    else:
        P21 = p2p_21

    A_mat = (w_smooth/w_couple) * mesh2.W + mesh2.A + (w_bij/w_couple) * P12.T @ mesh1.A @ P12
    B_mat = mesh2.A @ P21 @ mesh1.vertlist + (w_bij/w_couple) * P12.T @ mesh1.A @ mesh1.vertlist

    Y21 = sparse.linalg.spsolve(A_mat, B_mat)

    return Y21


def solve_Y21_arap_couple(mesh1, mesh2, p2p_21, params, pb_or_def='pull-back'):
    if pb_or_def == 'deformation':
        raise ValueError("Not implemented")

    smooth_weight = params['smooth_weight']
    couple_weight = params['couple_weight']

    vert_source = mesh2.vertlist
    new_vert1 = mesh1.vertlist[p2p_21]

    W_coo = arap.get_cotan_mat(mesh2.W)

    covariances = arap.get_covariances(W_coo, vert_source, new_vert1)
    rotations = arap.rotation_from_covariances(covariances)

    # Solve for positions
    rotated_edges = arap.get_rotated_lap(W_coo, rotations, vert_source)

    A_mat = smooth_weight / couple_weight * mesh2.W + sparse.eye(mesh2.n_vertices)
    B_mat = smooth_weight / couple_weight * rotated_edges + new_vert1

    Y21 = sparse.linalg.spsolve(A_mat, B_mat)

    return Y21
