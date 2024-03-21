import numpy as np
import scipy.linalg

from omegaconf import OmegaConf

import pyFM.spectral as spectral

# necessary_keys = ["couple_weight", "bij_weight"]

def orthogonalize_FM(FM_12):
    """
    Orthogonalize the given functional map

    Parameters
    ----------
    FM_12 : np.ndarray
        Functional map from mesh1 to mesh2
    
    Returns
    -------
    FM_12_orth : np.ndarray
        Orthogonalized functional map
    """
    k2, k1 = FM_12.shape
    U, _, VT = scipy.linalg.svd(FM_12)
    return U @ np.eye(k2, k1) @ VT


def _get_FM_type_from_params(params):
    """
    Get the functional map computation type from the given parameters

    Parameters
    ----------
    params : OmegaConf DictConfig
        Parameters dictionary

    Returns
    -------
    FM_type : str
        Functional map computation type
    """

    FM_type = OmegaConf.select(params, "meta_info.FM_type")
    if FM_type is not None:
        return FM_type
    
    if params.get("bij_weight", 0) > 0:
        if params.get("simple_energy", False):
            return "bijective_standard"
        return "bijective_weighted"

    return "single_standard"   


def _compute_bijective_FM_12(mesh1, mesh2, k, p2p_21, p2p_12):
    """
    Compute functional map using bijective energy, that is

    ||Phi_2 C_12 - Pi_21 Phi_1 ||_S2^2 + ||Pi_12 Phi_2 C_12 - Phi_1 ||_S1^2 

    where C_12 is the functional map, Pi_21 and Pi_12 are the pointwise maps in matrix form,
    Phi_1 and Phi_2 are the eigenvectors of the Laplacian of the two meshes, and ||.||_S1 and ||.||_S2
    denote the area-weighted L2 norm on the two meshes.

    Parameters
    ----------
    mesh1 : pyFM.mesh.TriMesh
        First mesh
    mesh2 : pyFM.mesh.TriMesh
        Second mesh
    k : int
        Number of eigenvectors to use
    p2p_21 : np.ndarray
        Pointwise map from mesh2 to mesh1
    p2p_12 : np.ndarray
        Pointwise map from mesh1 to mesh2

    Returns
    -------
    FM_12 : np.ndarray
        Functional map from mesh1 to mesh2
    """
    sub1 = np.concatenate([np.arange(mesh1.n_vertices), p2p_21])
    sub2 = np.concatenate([p2p_12, np.arange(mesh2.n_vertices)])

    FM_12 = spectral.mesh_p2p_to_FM(np.arange(sub2.size), mesh1, mesh2,
                                    dims=int(k), subsample=(sub1,sub2))

    return FM_12


def _compute_weighted_bijective_FM_12(mesh1, mesh2, k, p2p_21, p2p_12, bij_weight):
    """
    Comute functional map using the weighted bijective energy, that is

    ||Phi_2 C_12 - Pi_21 Phi_1 ||_S2^2 + w_bij * ||Pi_12 Phi_2 C_12 - Phi_1 ||_S1^2

    where C_12 is the functional map, Pi_21 and Pi_12 are the pointwise maps in matrix form,
    Phi_1 and Phi_2 are the eigenvectors of the Laplacian of the two meshes, and ||.||_S1 and ||.||_S2
    denote the area-weighted L2 norm on the two meshes.

    w_bij is the weight of the bijective term.

    Parameters
    ----------
    mesh1 : pyFM.mesh.TriMesh
        First mesh
    mesh2 : pyFM.mesh.TriMesh
        Second mesh
    k : int
        Number of eigenvectors to use
    p2p_21 : np.ndarray
        Pointwise map from mesh2 to mesh1
    p2p_12 : np.ndarray
        Pointwise map from mesh1 to mesh2
    bij_weight : float
        Weight of the bijective term
    
    Returns
    -------
    FM_12 : np.ndarray
        Functional map from mesh1 to mesh2
    """

    ev1 = mesh1.eigenvectors[:, :k]
    ev2 = mesh2.eigenvectors[:, :k]
    ev1_pb = mesh1.eigenvectors[p2p_21, :k] if p2p_21.ndim == 1 else p2p_21 @ mesh1.eigenvectors[:, :k]
    ev2_pb = mesh2.eigenvectors[p2p_12, :k] if p2p_12.ndim == 1 else p2p_12 @ mesh2.eigenvectors[:, :k]

    A_mat = np.eye(k) + bij_weight * ev2_pb.T @ mesh1.A @ ev2_pb
    B_mat = ev2.T @ mesh2.A @ ev1_pb + bij_weight * ev2_pb.T @ mesh1.A @ ev1

    FM_12 = np.linalg.solve(A_mat, B_mat)

    return FM_12


def solve_FM_12(mesh1, mesh2, k, p2p_21, p2p_12=None, params=None):
    """
    Solve for the functional map from mesh1 to mesh2 using the given parameters

    Parameters
    ----------
    mesh1 : pyFM.mesh.TriMesh
        First mesh
    mesh2 : pyFM.mesh.TriMesh
        Second mesh
    
    """
    FM_type = _get_FM_type_from_params(params)

    # Standard unweighted bijective FM computation
    if FM_type == "bijective_standard":
        FM_12 = _compute_bijective_FM_12(mesh1, mesh2, k, p2p_21, p2p_12)

    # Weighted bijective FM computation
    elif FM_type == "bijective_weighted":
        couple_weight = params['couple_weight']
        bij_weight = params['bij_weight']
        FM_12 = _compute_weighted_bijective_FM_12(mesh1, mesh2, k, p2p_21, p2p_12, bij_weight/couple_weight)

    # Standard pointwise to functional map conversion
    elif FM_type == "single_standard":
        FM_12 = spectral.mesh_p2p_to_FM(p2p_21, mesh1, mesh2, dims=k)
    
    else:
        raise NotImplementedError(f"Unknown FM computation type: {FM_type}")


    if params['orthogonal_FM']:
        FM_12 = orthogonalize_FM(FM_12)

    return FM_12