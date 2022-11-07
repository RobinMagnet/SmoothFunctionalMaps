import numpy as np


def coupling_emb(FM_12, mesh1, mesh2, normalize=False):
    """
    """
    k2, k1 = FM_12.shape

    evects1 = mesh1.eigenvectors[:, :k1].copy()  # (n1, k1)
    evects2 = mesh2.eigenvectors[:, :k2]  # (n2, k2)

    term1 = evects1  # (n1, k2)
    term2 = evects2 @ FM_12  # (n2, k2)

    if normalize:
        term1 /= np.sqrt(k2)
        term2 /= np.sqrt(k2)

    return term1, term2


def orthogonal_emb(FM_12, mesh1, mesh2, normalize=False):
    """
    """
    k2, k1 = FM_12.shape

    evects1 = mesh1.eigenvectors[:, :k1]  # (n1, k1)
    evects2 = mesh2.eigenvectors[:, :k2]  # (n2, k2)

    term1 = evects1 @ FM_12.T  # (n1, k2)
    term2 = evects2  # (n2, k2)

    if normalize:
        term1 = term1 / np.sqrt(k2)
        term2 = term2 / np.sqrt(k2)

    return term1, term2


def bijectivity_emb(FM_12, FM_21, mesh1, mesh2, normalize=False):
    """
    Compute terms A1, B1, A2, 2 so that the term :

        ||FM_12 @ FM_21 - I_2||^2 + ||FM_21 @ FM_12 - I_1||^2      (1)

    (with L1 and L2 diagonal matrices of LBO eigenvalues)
    will lead to an optimization problem2 of the form

                min_{T_21} ||T_21 A1 - B1||^2      (2)

                min_{T_12} ||T_12 A2 - B2||^2      (3)

    which can later be solved by  ```knn_query(A1, B1)```
    and ```knn_query(A2, B2)```

    Parameters:
    -------------------------
    FM_12  : (k2, k1) functional map from mesh1 to mesh2
    FM_21  : (k1, k2) functional map from mesh1 to mesh2
    mesh1  : TriMesh - source mesh for the functional map F_12
    mesh2  : TriMesh - target mesh for the functional map F_12


    Output:
    -------------------------
    A1      : (n1, k2) term A in (2)
    B1      : (n2, k2) term B in (2)
    A2      : (n2, k1) term A in (3)
    B2      : (n1, k1) term B in (3)
    """

    k2, k1 = FM_12.shape

    evects1 = mesh1.eigenvectors[:, :k1]  # (n1, k1)
    evects2 = mesh2.eigenvectors[:, :k2].copy()  # (n2, k2)

    # term1_21 = evects1 @ FM_21  # (n1, k2)
    # term2_21 = evects2  # (n2, k2)

    term1 = evects1 @ FM_21  # (n1, k2)
    term2 = evects2  # (n2, k2)

    if normalize:
        term1 /= np.sqrt(k2)
        term2 /= np.sqrt(k2)

    # term1_12 = evects2 @ FM_12  # (n2, k1)
    # term2_12 = evects1  # (n1, k1)

    # return term1_21, term2_21  #, term1_12, term2_12

    return term1, term2


def conformal_emb(FM_12, mesh1, mesh2, normalize=False):
    """
    Compute terms A and B so that the term :

                    ||C @ L1 @ C.T - L2||^2      (1)

    (with L1 and L2 diagonal matrices of LBO eigenvalues)
    will lead to an optimization problem of the form

                min_{T_21} ||T_21 A - B||^2      (2)

    which can later be solved by  ```knn_query(A, B)```

    Parameters:
    -------------------------
    FM     : (k2, k1) functional map from mesh1 to mesh2
    mesh1  : TriMesh - source mesh for the functional map
    mesh2  : TriMesh - target mesh for the functional map
    log    : bool - whether to output the original energy value (1)


    Output:
    -------------------------
    A      : (n1, k2) term A in (2)
    B      : (n2, k2) term B in (2)
    energy : value of (1)
    """

    k2, k1 = FM_12.shape

    evects1 = mesh1.eigenvectors[:, :k1]  # (n1, k1)
    evects2 = mesh2.eigenvectors[:, :k2]  # (n2, k2)

    evals1 = mesh1.eigenvalues[:k1]  # (k1,)
    evals2 = mesh2.eigenvalues[:k2]  # (k2,)

    term1 = evects1 @ (evals1[:, None] * FM_12.T)  # (n1, k2)
    term2 = evects2 * evals2[None, :]  # (n2, k2)

    if normalize:
        term1 /= np.sqrt(k2)
        term2 /= np.sqrt(k2)

    return term1, term2


def descriptor_emb(FM_12, descr1, descr2, mesh1, mesh2, normalize=False):
    """
    Compute terms A and B so that the term :

                      ||C @ F_1 - F_2||^2        (1)

    will lead to an optimization problem of the form

                min_{T_21} ||T_21 A - B||^2      (2)

    which can later be solved by  ```knn_query(A, B)```

    Parameters:
    -------------------------
    FM     : (k2, k1) functional map from mesh1 to mesh2
    descr1 : (k1, p) desriptors on shape 1 in the reduced basis
    descr2 : (k2, p) desriptors on shape 2 in the reduced basis
    mesh1  : TriMesh - source mesh for the functional map
    mesh2  : TriMesh - target mesh for the functional map
    log    : bool - whether to output the original energy value (1)


    Output:
    -------------------------
    A      : (n1, p) term A in (2)
    B      : (n2, p) term B in (2)
    energy : value of (1)
    """
    k2, k1 = FM_12.shape
    p = descr1.shape[1]

    evects1 = mesh1.eigenvectors[:, :k1]  # (n1, k1)
    evects2 = mesh2.eigenvectors[:, :k2]  # (n2, k2)

    term1 = evects1 @ descr1  # (n1, p)
    term2 = evects2 @ descr2  # (n2, p)

    if normalize:
        term1 /= np.sqrt(p)
        term2 /= np.sqrt(p)

    return term1, term2


def coupling_emb_spatial(Y_21, mesh1, mesh2, normalize=False):

    term1 = mesh1.vertlist.copy()
    term2 = Y_21.copy()

    if normalize:
        factor = np.sqrt(mesh2.l2_sqnorm(Y_21).sum())
        term1 /= factor
        term2 /= factor

    return term1, term2


def bijectivity_emb_spatial(Y_21, Y_12, mesh1, mesh2, normalize=False):

    term1 = Y_12.copy()
    term2 = mesh2.vertlist.copy()

    if normalize:
        factor = np.sqrt(mesh2.l2_sqnorm(Y_21).sum())
        term1 /= factor
        term2 /= factor

    return term1, term2
