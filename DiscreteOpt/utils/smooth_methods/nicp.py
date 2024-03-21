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
    """
    Computes M x G on eq (10) of the paper: "Optimal Step Nonrigid ICP Algorithms for Surface Registration" (Amberg et al.)
    """
    M = incidence_matrix_cotan(mesh1)

    stiffness_mat = sparse.kron(M, sparse.diags([1,1,1,gamma]), format='csr')

    return stiffness_mat


def correspondence_terms(source_positions, target_positions, weights=None):
    """
    Returns W @ D and W @ U from eq (9) of the paper: "Optimal Step Nonrigid ICP Algorithms for Surface Registration" (Amberg et al.)
    """
    vert1_ext = np.concatenate([source_positions, np.ones((source_positions.shape[0], 1))], axis=1)  # (n1, 4)
    term1 = sparse.block_diag(vert1_ext.tolist(), format='csr')  # (n1, 4 * n1)
    term2 = target_positions  # (n1, 3)

    if weights is not None:
        term1 = sparse.diags(weights) @ term1
        term2 = weights[:,None] * term2

    return term1, term2


def nicp_iteration(source_vertices, target_vertices, stiffness1, stiffness_weight=1, lmks_weight=10, landmarks=None, vertex_corr_weights=None):
    """
    Performs one iteration of the NICP algorithm
    
    Parameters
    ----------
    stiffness1 : scipy.sparse.csr_matrix (4*n1, 4*n1)
        Stiffness matrix for mesh1
    source_vertices : numpy.ndarray (n1, 3)
        Vertices of mesh1
    target_vertices : numpy.ndarray (n1, 3)
        Target vertices on mesh2
    stiffness_weight : float
        Weight for the stiffness term (alpha in the original paper)
    lmks_weight : float
        Weight for the landmarks term (beta in the original paper)
    landmarks : tuple of numpy.ndarray
        Landmarks for mesh1 and mesh2
    vertex_corr_weights : numpy.ndarray (n1,)
        Confidence weights for the correspondence of each vertex (default to 1)

    Returns
    -------
    X12 : numpy.ndarray (4*n1, 3)
        Transformation matrix
    transformed_verts : numpy.ndarray (n1, 3)
        Positions of the transformed vertices
    """
    corr1, corr2 = correspondence_terms(source_vertices, target_vertices, weights=vertex_corr_weights)  # (n1, 4*n1), (n1, 3)

    # WORKS ONLY IF NO WEIGHTS !!
    if landmarks is not None:
        lmks1, lmks2 = landmarks
        if vertex_corr_weights is not None:
            t1, t2 = correspondence_terms(source_vertices, target_vertices, weights=None)  # (n1, 4*n1), (n1, 3)
            lmks_term1 = t1[lmks1]  # (n_lmks, 4*e1)
            lmks_term2 = t2[lmks2]  # (n_lmks, 3)
        else:
            lmks_term1 = corr1[lmks1]  # (n_lmks, 4*e1)
            lmks_term2 = corr2[lmks2]  # (n_lmks, 3)

    # Solve the system
    if landmarks is not None:
        A_mat = stiffness_weight * stiffness1.T @ stiffness1 + corr1.T @ corr1 + lmks_weight * lmks_term1.T @ lmks_term1  # (4*n1, 4*n1)
        B_mat = corr1.T @ corr2 + lmks_weight * lmks_term1 .T @ lmks_term2  # (4*n1, 3)
    else:
        A_mat = stiffness_weight * stiffness1.T @ stiffness1 + corr1.T @ corr1  # (4*n1, 4*n1)
        B_mat = corr1.T @ corr2  # (4*n1, 3)

    X12 = sparse.linalg.spsolve(A_mat, B_mat)  # (4*n1, 3)

    if vertex_corr_weights is None:
        transformed_verts = corr1 @ X12
    elif landmarks is not None:
        transformed_verts = t1 @ X12
    else:
        vert1_ext = np.concatenate([source_vertices, np.ones((source_vertices.shape[0], 1))], axis=1)
        transformed_verts = sparse.block_diag(vert1_ext.tolist(), format='csr') @ X12

    return X12, transformed_verts


def solve_nicp_mesh(mesh1, mesh2, p2p_12=None, skew_weight=1, stiffness_weight=1e-2, lmks_weight=10, landmarks=None,
                    vertex_corr_weights=None, nit=10, n_jobs=1, use_cotan=False, update_corr=False, verbose=False):
    """
    Solves the NICP algorithm for two meshes.
    Only solves the linear system for the transformation matrix X12, and updates the correspondences at each step if update_corr is True.

    Parameters
    ----------
    mesh1 : trimesh.Trimesh
        Source mesh
    mesh2 : trimesh.Trimesh
        Target mesh
    p2p_12 : numpy.ndarray (n1,)
        Initial correspondences from mesh1 to mesh2. Can be None in the presence of landmarks
    skew_weight : float
        Weight for the skew term in the stiffness matrix
    stiffness_weight : float
        Weight for the stiffness term (alpha in the original paper)
    lmks_weight : float
        Weight for the landmarks term (beta in the original paper)
    landmarks : tuple of numpy.ndarray
        Landmarks for mesh1 and mesh2. Are indices of the vertices for each mesh
    vertex_corr_weights : numpy.ndarray (n1,)
        Confidence weights for the correspondence of each vertex (default to 1)
    nit : int
        Number of iterations
    n_jobs : int
        Number of parallel jobs for nn query
    use_cotan : bool
        Whether to use the cotan matrix for the stiffness matrix instead of the adjacency matrix
    update_corr : bool
        Whether to update the correspondences at each iteration
    verbose : bool
        Whether to show a progress bar

    Returns
    -------
    transformed_verts : numpy.ndarray (n1, 3)
        Positions of the transformed vertices

    """
    

    if p2p_12 is None and landmarks is None:
        raise ValueError("No guidance for the NICP algorithm. Either p2p_12 or landmarks must be provided.")
    
    # Original implementation uses graph laplacian
    if use_cotan:
        stiffness_mat = stiffness_matrix_cotan(mesh1, skew_weight)
    else:
        stiffness_mat = stiffness_matrix_graph(mesh1, skew_weight)

    # Initial guess for X12
    # X12 = np.tile(np.eye(3,4).T, (mesh1.n_vertices, 1))
        
    current_verts = mesh1.vertlist.copy()
    if p2p_12 is None:
        target_verts = mesh1.vertlist.copy()
    else:
        target_verts = mesh2.vertlist[p2p_12].copy() # (n1, 3)


    iterable = tqdm(range(nit)) if verbose else range(nit)
    for it in iterable:
        X12, transformed_verts = nicp_iteration(current_verts, target_verts, stiffness_mat, stiffness_weight=stiffness_weight,
                                  landmarks=landmarks, lmks_weight=lmks_weight, vertex_corr_weights=vertex_corr_weights)
        
        current_verts = transformed_verts.copy()
        if update_corr and it < nit - 1:
            p2p_12 = knn_query(mesh2.vertlist, transformed_verts, n_jobs=n_jobs)
            target_verts = mesh2.vertlist[p2p_12].copy()

    return transformed_verts