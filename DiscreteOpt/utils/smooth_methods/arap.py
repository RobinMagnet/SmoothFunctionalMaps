import numpy as np
import scipy.sparse as sparse
from tqdm.auto import tqdm

def solve_arap_mesh(mesh1, mesh2, p2p_12=None, landmarks=None, nit=10, verbose=False):
    """
    Solve the ARAP energy minimization for mesh1 to match mesh2.

    Parameters
    ----------
    mesh1 : trimesh.Trimesh
        source mesh
    mesh2 : trimesh.Trimesh
        target mesh
    p2p_12 : Optional - np.ndarray (n1,)
        point-to-point correspondences from mesh1 to mesh2. Used for initial guess, else use identity as initial guess.
    landmarks : Optional - Tuple[np.ndarray (n_lmks,), np.ndarray (n_lmks,)]
        indices of the landmarks in the source and target meshes
    nit : int
        number of iterations
    verbose : bool
        whether to display a progress bar

    Returns
    -------
    new_positions : np.ndarray (n1, 3)
        new positions of the vertices
    """
    use_landmarks = False
    if landmarks is not None:
        use_landmarks = True
        lmks1, lmks2 = landmarks

    W_coo = get_cotan_mat(mesh1.W)

    if use_landmarks:
        non_lmks_inds1 = np.array([i for i in range(mesh1.n_vertices) if i not in lmks1])
        solver_lap = sparse.linalg.factorized(mesh1.W[np.ix_(non_lmks_inds1, non_lmks_inds1)])
    else:
        solver_lap = sparse.linalg.factorized(mesh1.W)


    current_verts = mesh1.vertlist.copy()  # (n1, 3)
    if p2p_12 is None:
        target_verts = mesh1.vertlist.copy()
    else:
        target_verts = mesh2.vertlist[p2p_12] # (n1, 3)

    pbar = tqdm(range(nit)) if verbose else range(nit)
    for it in pbar:

        if use_landmarks:
            new_positions = solve_arap_iteration_with_lmks(W_coo, current_verts, target_verts, lmks1, mesh2.vertlist[lmks2], mesh1.W, non_lmks_inds1=non_lmks_inds1, solver_lap=solver_lap)
        else:
            new_positions = solve_arap_iteration_no_lmks(W_coo, current_verts, target_verts, W1=mesh1.W, solver_lap=solver_lap)

        target_verts = new_positions.copy()

    return new_positions


def solve_arap_mesh_coupling(mesh1, mesh2, p2p_12, landmarks=None, couple_weight=10, nit=10, verbose=False):
    """
    Solve the ARAP energy minimization for mesh1 to match mesh2 with coupling to some correspondences.

    Parameters
    ----------
    mesh1 : trimesh.Trimesh
        source mesh
    mesh2 : trimesh.Trimesh
        target mesh
    p2p_12 : np.ndarray (n1,)
        point-to-point correspondences from mesh1 to mesh2. Used for initial guess and for coupling.
    landmarks : Optional - Tuple[np.ndarray (n_lmks,), np.ndarray (n_lmks,)]
        indices of the landmarks in the source and target meshes
    couple_weight : float
        weight of the coupling term
    nit : int
        number of iterations
    verbose : bool
        whether to display a progress bar

    Returns
    -------
    new_positions : np.ndarray (n1, 3)
        new positions of the vertices
    """
    use_landmarks = False
    if landmarks is not None:
        use_landmarks = True
        lmks1, lmks2 = landmarks

    W_coo = get_cotan_mat(mesh1.W)

    if use_landmarks:
        non_lmks_inds1 = np.array([i for i in range(mesh1.n_vertices) if i not in lmks1])
        solver_lap = sparse.linalg.factorized(mesh1.W[np.ix_(non_lmks_inds1, non_lmks_inds1)] + couple_weight * sparse.eye(non_lmks_inds1.size))
    else:
        solver_lap = sparse.linalg.factorized(mesh1.W + couple_weight * sparse.eye(mesh1.n_vertices))

    current_verts = mesh1.vertlist.copy()  # (n1, 3)
    if p2p_12 is None:
        target_verts = mesh1.vertlist.copy()
    else:
        target_verts = mesh2.vertlist[p2p_12] # (n1, 3)

    pbar = tqdm(range(nit)) if verbose else range(nit)
    for it in pbar:

        if use_landmarks:
            new_positions = solve_arap_iteration_coupling_with_lmks(couple_weight, W_coo, current_verts, target_verts, lmks1, mesh2.vertlist[lmks2], verts_coupled=mesh2.vertlist[p2p_12], W1=mesh1.W, non_lmks_inds1=non_lmks_inds1, solver_lap=solver_lap)
        else:
            new_positions = solve_arap_iteration_coupling_no_lmks(couple_weight, W_coo, current_verts, target_verts, verts_coupled=mesh2.vertlist[p2p_12], W1=mesh1.W, solver_lap=solver_lap)

        target_verts = new_positions.copy()

    return new_positions


def solve_arap_iteration_with_lmks(W_coo, vert_source, first_guess, lmks_inds1, lmks_position, W1, non_lmks_inds1=None, solver_lap=None):
    """
    Solve one iteration of the ARAP energy minimization with landmarks preservation.

    Parameters
    ----------
    W_coo : sparse.coo_matrix
        sparse matrices with cotan weights (no diagonal)
    vert_source : np.ndarray (n1, 3)
        source vertices
    first_guess : np.ndarray (n1, 3)
        initial guess for the target vertices
    lmks_inds1 : np.ndarray (n_lmks,)
        indices of the landmarks in the source mesh
    lmks_position : np.ndarray (n_lmks, 3)
        positions of the landmarks in the target mesh
    W1 : sparse.csr_matrix
        sparse matrix with cotan weights (with diagonal)
    non_lmks_inds1 : Optional - np.ndarray (n1-n_lmks,) 
        indices of the non-landmarks in the source mesh
    solver_lap : callable
        Prefactorized solver for the laplacian system W1[non_lmks_inds1, non_lmks_inds1]

    Returns
    -------
    new_positions : np.ndarray (n1, 3)
        new positions of the vertices
    """
    if non_lmks_inds1 is None:
        non_lmks_inds1 = np.array([i for i in range(vert_source.shape[0]) if i not in lmks_inds1])

    covariances = get_covariances(W_coo, vert_source, first_guess)
    rotations = rotation_from_covariances(covariances)
    b_term = get_rotated_lap(W_coo, rotations, vert_source)

    new_positions = np.zeros_like(vert_source)

    right_hand_side = b_term[non_lmks_inds1] - W1[np.ix_(non_lmks_inds1, lmks_inds1)] @ lmks_position
    if solver_lap is not None:
        new_positions[non_lmks_inds1] = solver_lap(right_hand_side)
    else:
        new_positions[non_lmks_inds1] = sparse.linalg.spsolve(W1[np.ix_(non_lmks_inds1, non_lmks_inds1)], right_hand_side)

    new_positions[lmks_inds1] = lmks_position

    return new_positions


def solve_arap_iteration_no_lmks(W_coo, vert_source, first_guess, W1=None, solver_lap=None):
    """
    Solve one iteration of the ARAP energy minimization without landmarks preservation.
    You must provide either W1 or solver_lap.

    Parameters
    ----------
    W_coo : sparse.coo_matrix
        sparse matrices with cotan weights (no diagonal)
    vert_source : np.ndarray (n1, 3)
        source vertices
    first_guess : np.ndarray (n1, 3)
        initial guess for the target vertices
    W1 : sparse.csr_matrix
        sparse matrix with cotan weights (with diagonal)
    solver_lap : callable
        Prefactorized solver for the laplacian system W1

    Returns
    -------
    new_positions : np.ndarray (n1, 3)
        new positions of the vertices
    """
    if W1 is None and solver_lap is None:
        raise ValueError("Either W1 or solver_lap must be provided")
    
    covariances = get_covariances(W_coo, vert_source, first_guess)
    rotations = rotation_from_covariances(covariances)
    b_term = get_rotated_lap(W_coo, rotations, vert_source)

    if solver_lap is not None:
        new_positions = solver_lap(b_term)
    else:
        new_positions = sparse.linalg.spsolve(W1, b_term)

    return new_positions


def solve_arap_iteration_coupling_no_lmks(couple_weight, W_coo, vert_source, first_guess, verts_coupled=None, W1=None, solver_lap=None):
    """
    Solve one iteration of the ARAP energy minimization with coupling to some correspondences without landmarks preservation.
    ARAP term has weight 1.

    You must provide either W1 or solver_lap.

    Parameters
    ----------
    couple_weight : float
        weight of the coupling term
    W_coo : sparse.coo_matrix
        sparse matrices with cotan weights (no diagonal)
    vert_source : np.ndarray (n1, 3)
        source vertices
    first_guess : np.ndarray (n1, 3)
        initial guess for the target vertices
    verts_coupled : Optional - np.ndarray (n1, 3)
        initial guess for the target vertices. If None, it is set to first_guess
    W1 : sparse.csr_matrix
        sparse matrix with cotan weights (with diagonal)
    solver_lap : callable
        Prefactorized solver for the laplacian system W1 + couple_weight * I

    Returns
    -------
    new_positions : np.ndarray (n1, 3)
        new positions of the vertices
    """
    if W1 is None and solver_lap is None:
        raise ValueError("Either W1 or solver_lap must be provided")

    if verts_coupled is None:
        verts_coupled = first_guess

    covariances = get_covariances(W_coo, vert_source, first_guess)
    rotations = rotation_from_covariances(covariances)
    b_term = get_rotated_lap(W_coo, rotations, vert_source)

    right_hand_side = b_term + couple_weight * verts_coupled

    if solver_lap is not None:
        new_positions = solver_lap(right_hand_side)
    else:
        A_mat = W1 + couple_weight * sparse.eye(vert_source.shape[0])
        new_positions = sparse.linalg.spsolve(A_mat, right_hand_side)
    
    return new_positions


def solve_arap_iteration_coupling_with_lmks(couple_weight, W_coo, vert_source, first_guess, lmks_inds1, lmks_position, verts_coupled=None, W1=None, non_lmks_inds1=None, solver_lap=None):
    """
    Solve one iteration of the ARAP energy minimization with coupling to some correspondences and landmarks preservation.
    ARAP term has weight 1.

    You must provide either W1 or solver_lap.

    Parameters
    ----------
    couple_weight : float
        weight of the coupling term
    W_coo : sparse.coo_matrix
        sparse matrices with cotan weights (no diagonal)
    vert_source : np.ndarray (n1, 3)
        source vertices
    first_guess : np.ndarray (n1, 3)
        initial guess for the target vertices
    lmks_inds1 : np.ndarray (n_lmks,)
        indices of the landmarks in the source mesh
    lmks_position : np.ndarray (n_lmks, 3)
        positions of the landmarks in the target mesh
    verts_coupled : Optional - np.ndarray (n1, 3)
        initial guess for the target vertices. If None, it is set to first_guess
    W1 : sparse.csr_matrix
        sparse matrix with cotan weights (with diagonal)
    non_lmks_inds1 : Optional - np.ndarray (n1-n_lmks,) 
        indices of the non-landmarks in the source mesh
    solver_lap : callable
        Prefactorized solver for the laplacian system (W1 + couple_weight * I)[non_lmks_inds1, non_lmks_inds1]

    Returns
    -------
    new_positions : np.ndarray (n1, 3)
        new positions of the vertices
    """
    if W1 is None :
        raise ValueError("W1 must be provided in the presence of landmarks")

    if verts_coupled is None:
        verts_coupled = first_guess

    if non_lmks_inds1 is None:
        non_lmks_inds1 = np.array([i for i in range(vert_source.shape[0]) if i not in lmks_inds1])

    covariances = get_covariances(W_coo, vert_source, first_guess)
    rotations = rotation_from_covariances(covariances)
    b_term = get_rotated_lap(W_coo, rotations, vert_source)

    new_positions = np.zeros_like(vert_source)

    right_hand_side = b_term[non_lmks_inds1] + couple_weight * verts_coupled[non_lmks_inds1] - W1[np.ix_(non_lmks_inds1, lmks_inds1)] @ lmks_position
    if solver_lap is not None:
        new_positions[non_lmks_inds1] = solver_lap(right_hand_side)
    else:
        A_mat = W1[np.ix_(non_lmks_inds1, non_lmks_inds1)] + couple_weight * sparse.eye(non_lmks_inds1.size)
        new_positions[non_lmks_inds1] = sparse.linalg.spsolve(A_mat, right_hand_side)
    
    new_positions[lmks_inds1] = lmks_position

    return new_positions

### ARAP UTILS FUNCTIONS ###
def get_rotated_lap(W_coo, rotations, vertices):
    """
    Computes the rightmost term of the ARAP energy, which is the rotated laplacian.
    That is the right term of eq (8) and (9) in the paper:
    "As-Rigid-As-Possible Surface Modeling" by Olga Sorkine and Marc Alexa

    Output is a vector of size (N,3) with entry i being:

                    sum_{j in N(i)} w_ij/2 (R_i + R_j) (v_i - v_j)
    
    """
    I, J = W_coo.row, W_coo.col  # (2*n_edges,)
    w_vals = W_coo.data  # (2*n_edges,)

    # Compute w_ij/2 (R_i + R_j)(v_i - v_j) for each halfedge (i,j) and (j,i)
    rots = rotations[I] + rotations[J]  # (2*n_edges,3,3)
    edges = w_vals[:,None]/2 * (vertices[I] - vertices[J])  # (2*n_edges,3)

    V = np.einsum('mkp, mp -> mk', rots,  edges)  # (2*n_edges, 3)

    # We want to sum the results per-neihboring information
    In = np.repeat(I,3)
    Jn = np.tile(np.arange(3), I.size)
    Vn = V.flatten()

    b_mat = sparse.coo_matrix((Vn, (In,Jn)), shape=(vertices.shape[0], 3))

    b_mat = np.asarray(b_mat.todense())

    return b_mat


def get_cotan_mat(W):
    ## Compute the cotan matrix without diagonal
    W_coo = sparse.diags(W.diagonal()) - W
    W_coo.eliminate_zeros()

    return W_coo.tocoo()


def get_covariances(W_coo, vertices, new_vertices):
    """
    Compute covariances, so Eq (5) in the paper:
    "As-Rigid-As-Possible Surface Modeling" by Olga Sorkine and Marc Alexa

    Output is a matrix of size (N,3,3) with entry i being:
    
                        sum_{j in N(i)} w_ij (v_i - v_j) (v'_i - v'_j)^T
    """
    I, J = W_coo.row, W_coo.col  # (2*n_edges,)
    w_vals = W_coo.data  # (2*n_edges,)

    edges_base = vertices[I] - vertices[J]  # (2*n_edges, 3)
    edges_new = new_vertices[I] - new_vertices[J]  # (2*n_edges, 3)
    
    # w_ij (v_i - v_j) (v'_i - v'_j)^T for each halfedge (i,j) and (j,i)
    V = w_vals[:, None, None] * np.einsum('ni,nj->nij', edges_base, edges_new)  # (2*n_edges, 3, 3)


    # We now want to sum each halfedge (i,j) per-vertex i so in an (N,9) matrix

    In = np.repeat(I, 9)  # (18*n_edges)
    Jn = np.tile(np.arange(9), I.size)
    
    # We flatten in two steps to get the right order
    Vn = V.reshape(-1, 9).flatten()

    covariances = sparse.csr_matrix((Vn, (In, Jn)), shape=(vertices.shape[0], 9))

    covariances = np.asarray(covariances.todense()).reshape((vertices.shape[0],3,3))

    return covariances


def rotation_from_covariances(covariances):
    # U, _, VT = torch.linalg.svd(torch.tensor(covariances, dtype=torch.double))
    U, _, VT = np.linalg.svd(covariances)

    rotations = VT.transpose(0, 2, 1) @ U.transpose(0, 2, 1)
    U[np.linalg.det(rotations) < 0, :, -1] *= -1

    rotations = VT.transpose(0, 2, 1) @ U.transpose(0, 2, 1)

    return rotations
    # rotations = VT.transpose(1,2) @ U.transpose(1,2)

    # U[torch.where((torch.det(rotations) < 0))[0], :, -1] *= -1

    # rotations = VT.transpose(1,2) @ U.transpose(1,2)

    # return rotations.numpy()



