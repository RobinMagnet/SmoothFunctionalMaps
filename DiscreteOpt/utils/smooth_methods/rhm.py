import numpy as np
import scipy.sparse as sparse

def solve_bijective_dirichlet_mesh(mesh1, mesh2, P12, P21, couple_weight, bijective_weight, landmarks=None):
    use_landmarks = landmarks is not None
    if use_landmarks:
        lmks1, lmks2 = landmarks
        non_lmks_inds1 = [i for i in range(mesh1.n_vertices) if i not in lmks1]
        lmks_positions = mesh2.vertlist[lmks2]
    
    target_vertices = P12 @ mesh2.vertlist

    if use_landmarks:
        Y_12 = solve_bijective_dirichlet_lmks(target_vertices, lmks1, mesh2.vertlist, mesh1.A, mesh2.A, mesh1.W, P21, couple_weight, bijective_weight, lmks_position=lmks_positions, non_lmks_inds1=non_lmks_inds1)
    else:
        Y_12 = solve_bijective_dirichlet_no_lmks(target_vertices, mesh2.vertlist, mesh1.A, mesh2.A, mesh1.W, P21, couple_weight, bijective_weight)

    return Y_12


def solve_bijective_dirichlet_no_lmks(target_vertices, full_targets, A1, A2, W1, P_21, couple_weight, bijective_weight):
    """
    Solve the problem of finding optimal coordinates minimizing the Dirichlet energy while staying close to some correspondences

                \min_(Y_12)  ||Y_12||_{W_1}^2 + w_coupling || Y_12 - B ||_{A_1}^2 + w_bij * || P_21 Y_12 - X2||_{A2}

    Solved by solving setting the gradient to 0:
            
                    (W_1 + w_coupling A_1 + w_bij * P_21^T A2 P_21) Y = w_coupling A_1 B + w_bij * P_21^T A2 X2

    Parameters
    ----------
    target_vertices : numpy.ndarray (n1, 3)
        Target vertices on mesh2 to couple with
    full_targets : numpy.ndarray (n2, 3)
        Vertices on mesh2
    A1 : scipy.sparse.csr_matrix (n1, n1)
        Area matrix for mesh1
    A2 : scipy.sparse.csr_matrix (n2, n2)
        Area matrix for mesh2
    W1 : scipy.sparse.csr_matrix (n1, n1)
        Stiffness matrix of the laplacian (cotan weights) for mesh1
    P_21 : scipy.sparse.csr_matrix (n2, n1)
        Correspondence matrix from mesh2 to mesh1
    couple_weight : float
        Weight for the coupling term
    bijective_weight : float
        Weight for the bijective term

    Returns
    -------
    Y_12 : numpy.ndarray (n1, 3)
        Estimated positions of the vertices of mesh1 on mesh2
    """
    right_hand_side = couple_weight * (A1 @ target_vertices) + bijective_weight * (P_21.T @ A2 @ full_targets)
    left_hand_side = W1 + couple_weight * A1 + bijective_weight * P_21.T @ A2 @ P_21

    Y_12 = sparse.linalg.spsolve(left_hand_side, right_hand_side)

    return Y_12

def solve_bijective_dirichlet_lmks(target_vertices, lmks_inds1, full_targets, A1, A2, W1, P_21, couple_weight, bijective_weight, lmks_position=None, non_lmks_inds1=None):
    """
    Solve the problem of finding optimal coordinates minimizing the Dirichlet energy while staying close to some correspondences and respecting landmarks

                \min_(Y_12)  ||Y_12||_{W_1}^2 + w_coupling || Y_12 - B ||_{A_1}^2 + w_bij * || P_21 Y_12 - X2||_{A2}      w/  landmarks  (Y_12[lmks] = lmks_position)

    Solved by solving setting the gradient to 0:
            
                    (W_1 + w_coupling A_1 + w_bij * P_21^T A2 P_21) Y = w_coupling A_1 B + w_bij * P_21^T A2 X2     w/ landmarks  (Y[lmks] = lmks_position)

    Parameters
    ----------
    target_vertices : numpy.ndarray (n1, 3)
        Target vertices on mesh2 to couple with
    lmks_inds1 : list of int
        Indices of the landmarks
    full_targets : numpy.ndarray (n2, 3)
        Vertices on mesh2
    A1 : scipy.sparse.csr_matrix (n1, n1)
        Area matrix for mesh1
    A2 : scipy.sparse.csr_matrix (n2, n2)
        Area matrix for mesh2
    W1 : scipy.sparse.csr_matrix (n1, n1)
        Stiffness matrix of the laplacian (cotan weights) for mesh1
    P_21 : scipy.sparse.csr_matrix (n2, n1)
        Correspondence matrix from mesh2 to mesh1
    couple_weight : float
        Weight for the coupling term
    bijective_weight : float
        Weight for the bijective term
    lmks_position : numpy.ndarray (n_lmks, 3)
        Positions of the landmarks
    non_lmks_inds1 : list of int
        Indices of the non-landmarks

    Returns
    -------
    Y_12 : numpy.ndarray (n1, 3)
        Estimated positions of the vertices of mesh1 on mesh2
    """
    if non_lmks_inds1 is None:
        non_lmks_inds1 = [i for i in range(A1.shape[0]) if i not in lmks_inds1]
    if lmks_position is None:
        lmks_position = target_vertices[lmks_inds1]

    left_hand_side_full = W1 + couple_weight * A1 + bijective_weight * P_21.T @ A2 @ P_21  # (n1, n1)

    right_hand_side_full = (couple_weight * (A1 @ target_vertices) + bijective_weight * (P_21.T @ A2 @ full_targets))  # (n1, 3)


    right_hand_side = right_hand_side_full[non_lmks_inds1] - left_hand_side_full[np.ix_(non_lmks_inds1, lmks_inds1)] @ lmks_position  # (n1-n_lmks, 3)
    left_hand_side = left_hand_side_full[np.ix_(non_lmks_inds1, non_lmks_inds1)]  # (n1-n_lmks, n1-n_lmks)
    
    Y_12 = np.zeros_like(target_vertices)  # (n1, 3)
    Y_12[non_lmks_inds1] = sparse.linalg.spsolve(left_hand_side, right_hand_side)  # (n1-n_lmks, 3)
    Y_12[lmks_inds1] = lmks_position

    return Y_12