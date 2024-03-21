import numpy as np
import scipy.sparse as sparse

def solve_dirichlet_coupling_mesh(mesh1, mesh2, p2p_12, couple_weight=1e2, landmarks=None, solver=None):
    """
    Solve the problem of finding optimal coordinates minimizing the Dirichlet energy while staying close to some correspondences with potential landmarks:

                \min_Y  ||Y||_{W_1}^2 + w_coupling || Y - B ||_{A_1}^2      w/  landmarks  (Y[lmks] = lmks_position)

    Parameters
    ----------
    mesh1 : Mesh
        Source mesh
    mesh2 : Mesh
        Target mesh
    p2p_12 : numpy.ndarray (n1, 3) or (n1,)
        Coupling correspondences from mesh1 to mesh2
    couple_weight : float
        Weight for the coupling term
    landmarks : tuple of numpy.ndarray
        Landmarks for mesh1 and mesh2
    solver : callable
        Solver for the linear system (W_1 + w_coupling A_1) X = Y or (W_1 + w_coupling A_1)[non_lmks, non_lmks] X = Y in the case of landmarks

    Returns
    -------
    Y12 : numpy.ndarray (n1, 3)
        Solution of the problem
    """
    use_landmarks = landmarks is not None
    if use_landmarks:
        lmks1, lmks2 = landmarks
        non_lmks_inds1 = [i for i in range(mesh1.n_vertices) if i not in lmks1]
        lmks_positions = mesh2.vertlist[lmks2]

    if p2p_12.ndim == 1:
        target_vertices = mesh2.vertlist[p2p_12]
    else:
        target_vertices = p2p_12 @ mesh2.vertlist
    
    if use_landmarks:
        Y12 = solve_dirichlet_coupling_lmks(target_vertices, lmks1, mesh1.A, couple_weight, mesh1.W, lmks_position=lmks_positions, non_lmks_inds1=non_lmks_inds1, solver=solver)
    else:
        Y12 = solve_dirichlet_coupling_no_lmks(target_vertices, mesh1.A, couple_weight, W1=mesh1.W, solver=solver)
    
    return Y12


def solve_dirichlet_coupling_no_lmks(target_vertices, A1, couple_weight, W1=None, solver=None):
    """
    Solve the problem of finding optimal coordinates minimizing the Dirichlet energy while staying close to some correspondences

                \min_Y  ||Y||_{W_1}^2 + w_coupling || Y - B ||_{A_1}^2

    Solved by solving setting the gradient to 0:
            
                    W_1 Y + w_coupling A_1 Y = w_coupling A_1 B

    Parameters
    ----------
    target_vertices : numpy.ndarray (n1, 3)
        Target vertices on mesh2
    A1 : scipy.sparse.csr_matrix (n1, n1)
        Area matrix for mesh1
    couple_weight : float
        Weight for the coupling term
    W1 : scipy.sparse.csr_matrix (n1, n1)
        Stiffness matrix of the laplacian (cotan weights)
    solver : callable
        Solver for the linear system (W_1 + w_coupling A_1) X = Y
    """
    if W1 is None and solver is None:
        raise ValueError("Either W1 or a solver must be provided")
    
    right_hand_side = couple_weight * (A1 @ target_vertices)

    if solver is None:
        Y = sparse.linalg.spsolve(W1 + couple_weight * A1, right_hand_side)
    else:
        Y = solver(right_hand_side)
    
    return Y


def solve_dirichlet_coupling_lmks(target_vertices, lmks_inds1, A1, couple_weight, W1, lmks_position=None, non_lmks_inds1=None, solver=None):
    """
    Solve the problem of finding optimal coordinates minimizing the Dirichlet energy while staying close to some correspondences and respecting landmarks

                \min_Y  ||Y||_{W_1}^2 + w_coupling || Y - B ||_{A_1}^2      w/  landmarks  (Y[lmks] = lmks_position)

    Solved by solving setting the gradient to 0:
            
                    W_1 Y + w_coupling A_1 Y = w_coupling A_1 B     w/ landmarks  (Y[lmks] = lmks_position)

    Parameters
    ----------
    target_vertices : numpy.ndarray (n1, 3)
        Target vertices on mesh2
    lmks_inds1 : list of int
        Indices of the landmarks
    A1 : scipy.sparse.csr_matrix (n1, n1)
        Area matrix for mesh1
    couple_weight : float
        Weight for the coupling term
    W1 : scipy.sparse.csr_matrix (n1, n1)
        Stiffness matrix of the laplacian (cotan weights)
    lmks_position : numpy.ndarray (n_lmks, 3)
        Position of the landmarks. If None, it is set to target_vertices[lmks_inds1]
    non_lmks_inds1 : list of int
        Indices of the non-landmarks vertices. If None, it is set to [i for i in range(A1.shape[0]) if i not in lmks_inds1]
    solver : callable
        Solver for the linear system (W_1 + w_coupling A_1)[non_lmks, non_lmks] X = Y

    Returns
    -------
    Y : numpy.ndarray (n1, 3)
        Solution of the problem
    """

    if W1 is None:
        raise ValueError("W1 must be provided")
    
    if non_lmks_inds1 is None:
        non_lmks_inds1 = [i for i in range(A1.shape[0]) if i not in lmks_inds1]  # (n1-n_lmks,)
    
    if lmks_position is None:
        lmks_position = target_vertices[lmks_inds1]  # (n_lmks, 3)
    
    Y = np.zeros_like(target_vertices)  # (n1, 3)

    right_hand_side = couple_weight * (A1 @ target_vertices)[non_lmks_inds1] - (W1 + couple_weight * A1)[np.ix_(non_lmks_inds1, lmks_inds1)] @ lmks_position  # (n1-n_lmks, 3)
    if solver is None:
        left_hand_side = W1[np.ix_(non_lmks_inds1, non_lmks_inds1)] + couple_weight * A1[np.ix_(non_lmks_inds1, non_lmks_inds1)]
        Y[non_lmks_inds1] = sparse.linalg.spsolve(left_hand_side, right_hand_side)  # (n1-n_lmks, 3)
    else:
        Y[non_lmks_inds1] = solver(right_hand_side)

    Y[lmks_inds1] = lmks_position

    return Y
