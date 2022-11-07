import numpy as np
import scipy.sparse as sparse
from tqdm.notebook import tqdm
from .nn_utils import knn_query
# import torch


def incidence_matrix(mesh1):
    edges = np.sort(mesh1.edges, axis=1)

    I = np.repeat(np.arange(edges.shape[0]), 2)
    J = edges.flatten()
    V = np.tile([-1, 1], edges.shape[0])

    M = sparse.csr_matrix((V, (I, J)), shape=(edges.shape[0], mesh1.n_vertices))

    return M


def stiffness_matrix_graph(mesh1, gamma):
    M = incidence_matrix(mesh1)
    stiffness_mat = sparse.kron(M, sparse.diags([1,1,1,gamma]), format='csr')

    return stiffness_mat


def edges_from_W(W):
    W_nodiag = (sparse.diags(W.diagonal()) - W).tocoo()

    edges = np.concatenate([W_nodiag.row[:,None], W_nodiag.col[:,None]], axis=1)

    edges = np.unique(np.sort(edges,axis=1),axis=0)

    return edges


def incidence_matrix_cotan(mesh1):
    # edges = np.sort(mesh1.edges, axis=1)
    edges = edges_from_W(mesh1.W)

    I = np.repeat(np.arange(edges.shape[0]), 2)
    J = edges.flatten()

    values = -np.array(mesh1.W[edges[:,0],edges[:,1]]).flatten()

    V = np.concatenate([-values[:,None], values[:,None]],axis=1).flatten()

    # V = np.tile([-1, 1], edges.shape[0])

    M = sparse.csr_matrix((V, (I, J)), shape=(edges.shape[0], mesh1.n_vertices))

    return M


def stiffness_matrix_cotan(mesh1, gamma):
    M = incidence_matrix_cotan(mesh1)

    stiffness_mat = sparse.kron(M, sparse.diags([1,1,1,gamma]), format='csr')

    return stiffness_mat


def correspondence_terms(vert1, vert2, p2p_12, weights=None):
    vert1_ext = np.concatenate([vert1, np.ones((vert1.shape[0], 1))], axis=1)
    term1 = sparse.block_diag(vert1_ext.tolist(), format='csr')
    term2 = vert2[p2p_12]  # (n,3)

    if weights is not None:
        term1 = sparse.diags(weights) @ term1
        term2 = weights[:,None] * term2

    return term1, term2


def nicp_iteration(stiffness1, vert1, vert2, p2p_12, alpha, lmks_weight=10, landmarks=None, x0=None, weights=None, n_jobs=1, verbose=False):
    corr1, corr2 = correspondence_terms(vert1, vert2, p2p_12, weights=weights)

    # WORKS ONLY IF NO WEIGHTS !!
    if landmarks is not None:
        lmks1, lmks2 = landmarks
        lmks_term1 = corr1[lmks1]  # (n_lmks, 4*e1)
        lmks_term2 = vert2[lmks2]  # (n_lmks, 3)

    if landmarks is not None:
        A_mat = alpha * stiffness1.T @ stiffness1 + corr1.T @ corr1 + lmks_weight * lmks_term1.T @ lmks_term1  # (4*n1, 4*n1)
        B_mat = corr1.T @ corr2 + lmks_weight * lmks_term1 .T @ lmks_term2  # (4*n1, 3)
    else:
        A_mat = alpha * stiffness1.T @ stiffness1 + corr1.T @ corr1
        B_mat = corr1.T @ corr2  # (4*n1, 3)

    X12 = sparse.linalg.spsolve(A_mat, B_mat)

    if weights is None:
        transformed_verts = corr1 @ X12
    else:
        vert1_ext = np.concatenate([vert1, np.ones((vert1.shape[0], 1))], axis=1)
        transformed_verts = sparse.block_diag(vert1_ext.tolist(), format='csr') @ X12

    p2p_12 = knn_query(vert2, transformed_verts, n_jobs=n_jobs)

    return p2p_12, X12, transformed_verts


def solve_nicp(mesh1, mesh2, p2p_12,
               gamma=1, alpha=1e-2, landmarks=None, lmks_weight=10,
               use_weighting=False, nit=10, n_jobs=1, use_cotan=False, verbose=False):
    if use_cotan:
        stiffness_mat = stiffness_matrix_cotan(mesh1, gamma)
    else:
        stiffness_mat = stiffness_matrix_graph(mesh1, gamma)

    X12 = np.tile(np.eye(3,4).T, (mesh1.n_vertices, 1))

    iterable = tqdm(range(nit)) if verbose else range(nit)
    for it in iterable:
        nicp_res = nicp_iteration(stiffness_mat, mesh1.vertlist, mesh2.vertlist, p2p_12, alpha,
                                  x0=X12, weights=None, landmarks=landmarks, lmks_weight=lmks_weight,
                                  n_jobs=n_jobs, verbose=verbose)

        p2p_12, X12, transformed_verts = nicp_res

    return p2p_12, transformed_verts