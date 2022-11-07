import numpy as np
import scipy.sparse as sparse

from . import nn_utils as utils


def get_rotated_lap(W_coo, rotations, vertices):
    I, J = W_coo.row, W_coo.col  # (2*n_edges,)
    w_vals = W_coo.data  # (2*n_edges,)

    rots = rotations[I] + rotations[J]  # (2*n_edges,3,3)
    edges = w_vals[:,None]/2 * (vertices[I] - vertices[J])  # (2*n_edges,3)

    V = np.einsum('mkp, mp -> mk', rots,  edges)  # (2*n_edges, 3)

    In = np.repeat(I,3)
    Jn = np.tile(np.arange(3), I.size)
    Vn = V.flatten()

    b_mat = sparse.coo_matrix((Vn, (In,Jn)), shape=(vertices.shape[0], 3))

    b_mat = np.asarray(b_mat.todense())

    return b_mat


def get_cotan_mat(W):
    W_coo = sparse.diags(W.diagonal()) - W
    W_coo.eliminate_zeros()

    return W_coo.tocoo()


def get_covariances(W_coo, vertices, new_vertices):
    I, J = W_coo.row, W_coo.col  # (2*n_edges,)
    w_vals = W_coo.data  # (2*n_edges,)

    edges_base = vertices[I] - vertices[J]  # (2*n_edges, 3)
    edges_new = new_vertices[I] - new_vertices[J]  # (2*n_edges, 3)

    V = w_vals[:, None, None] * np.einsum('ni,nj->nij', edges_base, edges_new)  # (2*n_edges, 3, 3)

    # return 3*I1, 3*I1+1, 3*I1+2, 3*I2, 3*I2+1, ...
    In = np.concatenate([3*I[:,None], 3*I[:,None]+1, 3*I[:,None]+2], axis=1).flatten()  # (6*n_edges)

    # return 3*I1, 3*I1, 3*I1, 3*I1+1, 3*I1+1,3*I1+1, 3*I1+2, 3*I1+2, 3*I1+2, 3*I1+2, 3*I2, 3*I2, 3*I2, ...
    In = np.repeat(In, 3)  # (18*n_edges)

    Jn = np.tile(np.arange(3), 3*I.size)
    Vn = V.flatten()

    covariances = sparse.csr_matrix((Vn, (In, Jn)), shape=(3*vertices.shape[0], 3))

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


# def arap_iteration(vert_source, W_coo, first_guess, landmarks=None):
#     covariances = get_covariances(W_coo, vert_source, first_guess)
#     rotations = rotation_from_covariances(covariances)


def solve_arap_mesh(mesh1, mesh2, p2p_12, landmarks=None, nit=10, n_inner=5, nn_update=True, n_jobs=1, verbose=False):
    use_landmarks = False
    if landmarks is not None:
        use_landmarks = True
        lmks1, lmks2 = landmarks

    W_coo = get_cotan_mat(mesh1.W)

    if use_landmarks:
        inds1 = np.array([i for i in range(mesh1.n_vertices) if i not in lmks1])
        inds2 = np.array([i for i in range(mesh2.n_vertices) if i not in lmks2])
        solver_lap = sparse.linalg.factorized(mesh1.W[np.ix_(inds1, inds1)])
        # solver_lap = sparse.linalg.factorized(mesh1.W[inds1])
        # print(mesh1.W[np.ix_(inds1, inds2)].shape)
    else:
        solver_lap = sparse.linalg.factorized(mesh1.W)

    current_vert = mesh1.vertlist.copy()

    new_vert1 = mesh2.vertlist[p2p_12]

    n_inner = n_inner if nn_update else 1
    for it in tqdm(range(nit)):

        iter_inner = tqdm(range(n_inner)) if (verbose and n_inner > 1) else range(n_inner)
        for it_inner in iter_inner:
            # Solve for rotations
            covariances = get_covariances(W_coo, current_vert, new_vert1)
            rotations = rotation_from_covariances(covariances)

            # Solve for positions
            # if use_landmarks:
            #     vert_adapted = new_vert1.copy()
            #     vert_adapted[lmks1] = mesh2.vertlist[lmks2]
            #     b_term = get_rotated_lap(W_coo, rotations, vert_adapted)
            # else:
            b_term = get_rotated_lap(W_coo, rotations, current_vert)

            if use_landmarks:
                new_positions = np.zeros_like(mesh1.vertlist)
                # new_positions[inds1] = solver_lap(b_term[inds1])
                new_positions[inds1] = solver_lap(b_term[inds1])
                new_positions[lmks1] = mesh2.vertlist[lmks2]
            else:
                new_positions = solver_lap(b_term)

            # R, t = utils.rigid_alignment(new_positions[lmks1], mesh2.vertlist[lmks2], p2p=np.arange(lmks1.shape[0]), return_deformed=False, return_params=True)
            # new_positions = new_positions@R.T + t

            # if use_landmarks:# and ((it<nit-1) or (it_inner<n_inner-1)):
            #     # new_positions += np.mean(mesh2.vertlist[lmks2] - new_positions[lmks1],axis=0,keepdims=True)
            #     new_positions[lmks1] = mesh2.vertlist[lmks2]

            # new_positions -= np.sum(mesh1.A@new_positions,axis=0, keepdims=True) / mesh1.area

            new_vert1 = new_positions.copy()
        if nn_update:
            p2p_12 = utils.knn_query(mesh2.vertlist, new_positions)
            new_vert1 = mesh2.vertlist[p2p_12]
        else:
            new_vert1 = new_positions

    return new_vert1


def solve_arap_mesh_new(mesh1, mesh2, p2p_12=None, landmarks=None, nit=10, n_inner=5, lmks_weight=50, nn_update=True, verbose=False):
    use_landmarks = False
    if landmarks is not None:
        use_landmarks = True
        lmks1, lmks2 = landmarks

    W_coo = get_cotan_mat(mesh1.W)

    if use_landmarks:
        inds1 = np.array([i for i in range(mesh1.n_vertices) if i not in lmks1])
        inds2 = np.array([i for i in range(mesh2.n_vertices) if i not in lmks2])
        # solver_lap = sparse.linalg.factorized(mesh1.W[np.ix_(inds1, inds1)])
    # solver_lap = sparse.linalg.factorized(mesh1.W[inds1])

    anchor_res = np.zeros((mesh1.n_vertices, 3))
    anchor_res[lmks1] = mesh2.vertlist[lmks2]

    e_diag = np.zeros(mesh1.n_vertices)
    e_diag[lmks1] = 1
    E_mat = sparse.diags(e_diag)

    solver_lap = sparse.linalg.factorized((mesh1.W.T @ mesh1.W + lmks_weight*E_mat.T@E_mat).tocsc())
    # print(mesh1.W[np.ix_(inds1, inds2)].shape)
    # else:
    #     solver_lap = sparse.linalg.factorized(mesh1.W)

    if p2p_12 is None:
        new_vert1 = mesh1.vertlist.copy()
    else:
        new_vert1 = mesh2.vertlist[p2p_12]

    current_vert = mesh1.vertlist.copy()
    covariances = get_covariances(W_coo, current_vert, new_vert1)
    rotations = rotation_from_covariances(covariances)
    b_term = get_rotated_lap(W_coo, rotations, current_vert)

    new_positions = solver_lap(mesh1.W.T@b_term + lmks_weight*anchor_res)
    new_vert1 = new_positions.copy()
    # new_vert1 = mesh2.vertlist[p2p_12]

    n_inner = n_inner if nn_update else 1
    iterable = range(nit) if not verbose else tqdm(range(nit))
    for it in iterable:

        iter_inner = tqdm(range(n_inner)) if (verbose and n_inner > 1) else range(n_inner)
        for it_inner in iter_inner:
            # Solve for rotations
            covariances = get_covariances(W_coo, current_vert, new_vert1)
            rotations = rotation_from_covariances(covariances)

            new_positions = solver_lap(mesh1.W.T@b_term + lmks_weight*anchor_res)

            new_vert1 = new_positions.copy()

    return new_vert1


def solve_arap_simple(vert_source, W_mat, first_guess, landmarks=None, nit=10, verbose=False):
    use_landmarks = False
    if landmarks is not None:
        use_landmarks = True
        lmks1, lmks2 = landmarks

    W_coo = get_cotan_mat(W_mat)

    solver_lap = sparse.linalg.factorized(W_mat)

    new_vert1 = first_guess
    iterable = tqdm(range(nit)) if verbose else range(nit)
    for it in iterable:

        # Solve for rotations
        covariances = get_covariances(W_coo, vert_source, new_vert1)
        rotations = rotation_from_covariances(covariances)

        # Solve for positions
        b_term = get_rotated_lap(W_coo, rotations, vert_source)
        new_positions = solver_lap(b_term)

        R, t = utils.rigid_alignment(new_positions[lmks1], lmks2, p2p=np.arange(lmks1.shape[0]), return_deformed=False, return_params=True)
        new_positions = new_positions@R.T + t

        if use_landmarks:
            # new_positions += np.mean(lmks2 - new_positions[lmks1],axis=0,keepdims=True)
            new_positions[lmks1] = lmks2

        new_vert1 = new_positions

    return new_positions