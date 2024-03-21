import numpy as np
from omegaconf import OmegaConf

from . import embeddings as emb

from pyFM.spectral import projection_utils as pju
from pyFM.spectral import nn_utils

# necessary_keys = ["normalize_terms",
#                   "couple_weight", # if 'coupling' in emb_type
#                     "bij_weight", # if 'bijectivity' in emb_type
#                     "conf_weight", # if 'conformal' in emb_type
#                     "descr_weight", # if 'descriptors' in emb_type
#                     "global_reweight", # if 'global_reweight' not necessary
#                     "orth_coupling", # if orthogonal coupling
#                     "normalize_terms", # if normalize terms
# ]

def _scale_emb(embeddings, factor):
    """
    Scale the given embeddings by the given factor

    Parameters
    ----------
    embeddings : tuple[np.ndarray, np.ndarray]
        Embeddings to scale
    factor : float
        Scaling factor

    Returns
    -------
    emb1 : np.ndarray
        Scaled embedding 1
    emb2 : np.ndarray
        Scaled embedding 2
    """

    emb1, emb2 = embeddings

    if factor == 1:
        return emb1, emb2

    return factor * emb1, factor * emb2

def _get_primal_embedding_types_from_params(params) -> list[str]:
    """
    Get the primal embedding types from the given parameters

    Parameters
    ----------
    params : OmegaConf DictConfig
        Parameters dictionary

    Returns
    -------
    emb_types : list[str]
        List of primal embedding types
    """
    
    emb_types = OmegaConf.select(params, "meta_info.emb_types")
    if emb_types is not None:
        return emb_types
    
    emb_types = []
    if params.get('couple_weight', 0) > 0:
        emb_types.append("coupling")

    if params["method"].lower() == 'rhm':
        if not params.get('only_coupling_for_nn', False) and params.get('bij_weight', 0) > 0:
            emb_types.append("bijectivity")
    # elif params.get('bij_weight', 0) > 0:
    #     if not params.get('only_coupling_for_nn', False):
    #         emb_types.append("bijectivity")

    return emb_types

def _get_spectral_embedding_types_from_params(params) -> list[str]:
    """
    Get the spectral embedding types from the given parameters

    Parameters
    ----------
    params : OmegaConf DictConfig
        Parameters dictionary

    Returns
    -------
    emb_types : list[str]
        List of spectral embedding types
    """
    
    emb_types = OmegaConf.select(params, "meta_info.emb_types")
    if emb_types is not None:
        return emb_types
    
    emb_types = []
    if params.get("couple_weight", 0) > 0:
        if params.get("orth_coupling", False):
            emb_types.append("orth_coupling")
        emb_types.append("coupling")
    
    if params.get("bij_weight", 0) > 0:
        emb_types.append("bijectivity")
    
    if params.get("conf_weight", 0) > 0:
        emb_types.append("conformal")
    
    if params.get("descr_weight", 0) > 0:
        emb_types.append("descriptors")

    return emb_types

def _get_spectral_emb_dim_from_emb_types(emb_types: list[str]):
    """
    Get the embedding dimension from the given embedding types

    Parameters
    ----------
    emb_types : list[str]
        List of spectral embedding types

    Returns
    -------
    emb_dim : str
        Embedding dimension as a function of K1, K2 and p (descriptors)
    """
    emb_dim_K1 = 0
    emb_dim_K2 = 0
    emb_dim_p = 0
    emb_dim_bias = 0

    if "coupling" in emb_types:
        emb_dim_K1 += 1
    
    if "orth_coupling" in emb_types:
        emb_dim_K2 += 1
    
    if "bijectivity" in emb_types:
        emb_dim_K2 += 1
    
    if "conformal" in emb_types:
        emb_dim_K2 += 1
    
    if "descriptors" in emb_types:
        emb_dim_p += 1

    return emb_dim_K1, emb_dim_K2, emb_dim_p, emb_dim_bias


def solve_p2p_21_spectral(mesh1, mesh2, FM_12, FM_21=None, descr1=None, descr2=None, params=None, n_jobs=1, precise=False):
    """
    Solve for the pointwise map from mesh2 to mesh1 using spectral energies

    Parameters
    ----------
    mesh1 : pyFM.mesh.TriMesh
        First mesh
    mesh2 : pyFM.mesh.TriMesh
        Second mesh
    FM_12 : np.ndarray (k2, k1)
        Functional map from mesh1 to mesh2
    FM_21 : np.ndarray (k1, k2) Optional
        Functional map from mesh2 to mesh1. Only used if 'bijectivity' term is used
    descr1 : np.ndarray (n1, d) Optional
        Descriptors for mesh1. Only used if 'descriptors' term is used
    descr2 : np.ndarray (n2, d)
        Descriptors for mesh2. Only used if 'descriptors' term is used
    params : OmegaConf DictConfig
        Parameters dictionary
    n_jobs : int
        Number of jobs to use
    precise : bool
        Whether to use vertex to barycentric coordinates projection or nearest neighbor search
    
    Returns
    -------
    p2p_21 : np.ndarray or scipy.sparse.csr_matrix
        Pointwise map from mesh2 to mesh1
    """
    emb1, emb2 = get_spectral_embeddings_21(mesh1, mesh2, FM_12, FM_21=FM_21, descr1=descr1, descr2=descr2, params=params)

    if precise:
        p2p_21 = pju.project_pc_to_triangles(emb1, mesh1.facelist, emb2, n_jobs=n_jobs)  # (n2, n1) sparse
    else:
        p2p_21 = nn_utils.knn_query(emb1, emb2, n_jobs=n_jobs)  # (n2,)

    return p2p_21


def solve_p2p_21_with_primal(mesh1, mesh2, FM_12, Y_21, FM_21=None, Y_12=None, descr1=None, descr2=None, params_sp=None, params_sm=None, n_jobs=1, precise=False):
    emb1_sp, emb2_sp = get_spectral_embeddings_21(mesh1, mesh2, FM_12, FM_21=FM_21, descr1=descr1, descr2=descr2, params=params_sp)
    emb1_pr, emb2_pr = get_primal_embeddings_21(mesh1, mesh2, Y_21, Y_12=Y_12, params=params_sm)

    # emb1 = np.concatenate([emb1_sp, emb1_pr], axis=1)
    # emb2 = np.concatenate([emb2_sp, emb2_pr], axis=1)
    emb1 = np.concatenate([emb1_pr, emb1_sp], axis=1)
    emb2 = np.concatenate([emb2_pr, emb2_sp], axis=1)

    # print(np.linalg.norm(emb1_sp, axis=1).mean(), np.linalg.norm(emb1_pr, axis=1).mean())
    # return emb1, emb2

    if precise:
        p2p_21 = pju.project_pc_to_triangles(emb1, mesh1.facelist, emb2, n_jobs=n_jobs)
    else:
        p2p_21 = nn_utils.knn_query(emb1, emb2, n_jobs=n_jobs)

    return p2p_21



def get_spectral_embeddings_21(mesh1, mesh2, FM_12, FM_21=None, descr1=None, descr2=None, params=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the concatenation of the spectral embeddings for each shape in order to compute p2p_21

    Parameters
    ----------
    mesh1 : pyFM.mesh.TriMesh
        First mesh
    mesh2 : pyFM.mesh.TriMesh
        Second mesh
    FM_12 : np.ndarray (k2, k1)
        Functional map from mesh1 to mesh2
    FM_21 : np.ndarray (k1, k2) Optional
        Functional map from mesh2 to mesh1. Only used if 'bijectivity' term is used
    descr1 : np.ndarray (n1, d) Optional
        Descriptors for mesh1. Only used if 'descriptors' term is used
    descr2 : np.ndarray (n2, d) Optional
        Descriptors for mesh2. Only used if 'descriptors' term is used
    params : OmegaConf DictConfig
        Parameters dictionary
    """

    embedding_type_list = _get_spectral_embedding_types_from_params(params)
    
    emb_total1 = []
    emb_total2 = []

    if "orth_coupling" in embedding_type_list:
        emb_orth_couple = get_spectral_orth_coupling_embs(mesh1, mesh2, FM_12, params=params)  # (n1, k2), (n2, k2)
        emb_orth_couple = _scale_emb(emb_orth_couple, np.sqrt(params["couple_weight"]))
        emb_total1.append(emb_orth_couple[0])
        emb_total2.append(emb_orth_couple[1])
    
    if "coupling" in embedding_type_list:
        emb_couple = get_spectral_coupling_embs(mesh1, mesh2, FM_12, params=params)  # (n1, k1), (n2, k1)
        emb_couple = _scale_emb(emb_couple, np.sqrt(params["couple_weight"]))
        emb_total1.append(emb_couple[0])
        emb_total2.append(emb_couple[1])

    if "bijectivity" in embedding_type_list:
        emb_bij = get_spectral_bij_embs(mesh1, mesh2, FM_21, params=params)  # (n1, k2), (n2, k2)
        emb_bij = _scale_emb(emb_bij, np.sqrt(params["bij_weight"]))
        emb_total1.append(emb_bij[0])
        emb_total2.append(emb_bij[1])
    
    if "conformal" in embedding_type_list:
        emb_conf = get_spectral_conf_embs(mesh1, mesh2, FM_12, params=params)  # (n1, k2), (n2, k2)
        emb_conf = _scale_emb(emb_conf, np.sqrt(params["conf_weight"]))
        emb_total1.append(emb_conf[0])
        emb_total2.append(emb_conf[1])

    if "descriptors" in embedding_type_list:
        emb_descr = get_spectral_descr_embs(mesh1, mesh2, FM_12, descr1, descr2, params=params) # (n1,p), (n2, p)
        emb_descr = _scale_emb(emb_descr, np.sqrt(params["descr_weight"]))
        emb_total1.append(emb_descr[0])
        emb_total2.append(emb_descr[1])

    emb1 = np.concatenate(emb_total1, axis=1)
    emb2 = np.concatenate(emb_total2, axis=1)

    if "global_reweight" in params.keys() and params["global_reweight"] is not None and params["global_reweight"] != 1.0:
        emb1 = emb1 * params["global_reweight"]
        emb2 = emb2 * params["global_reweight"]

    return emb1, emb2


def get_primal_embeddings_21(mesh1, mesh2, Y_21, Y_12=None, params=None):
    """
    Compute the primal embeddings for the given shapes

    Parameters
    ----------
    mesh1 : pyFM.mesh.TriMesh
        First mesh
    mesh2 : pyFM.mesh.TriMesh
        Second mesh
    Y_21 : np.ndarray (n1, k)
        Primal map from mesh1 to mesh2
    Y_12 : np.ndarray (n2, k) Optional
        Primal map from mesh2 to mesh1. Only used if 'bijectivity' term is used
    params : OmegaConf DictConfig
        Parameters dictionary

    Returns
    -------
    emb1 : np.ndarray (n1, p)
        Primal embeddings for mesh1
    emb2 : np.ndarray (n2, p)
    """
    embedding_type_list = _get_primal_embedding_types_from_params(params)

    emb_total1 = []
    emb_total2 = []

    if "coupling" in embedding_type_list:
        emb_couple = get_primal_coupling_emb(mesh1, mesh2, Y_21, params=params)
        emb_couple = _scale_emb(emb_couple, np.sqrt(params["couple_weight"]))
        emb_total1.append(emb_couple[0])
        emb_total2.append(emb_couple[1])
    
    if "bijectivity" in embedding_type_list:
        emb_bij = get_primal_bij_emb(mesh1, mesh2, Y_21, Y_12, params=params)
        emb_bij = _scale_emb(emb_bij, np.sqrt(params["bij_weight"]))
        emb_total1.append(emb_bij[0])
        emb_total2.append(emb_bij[1])
    
    emb1 = np.concatenate(emb_total1, axis=1)
    emb2 = np.concatenate(emb_total2, axis=1)

    if "global_reweight" in params.keys() and params["global_reweight"] is not None and params["global_reweight"] != 1.0:
        emb1 = emb1 * params["global_reweight"]
        emb2 = emb2 * params["global_reweight"]

    return emb1, emb2


def get_spectral_coupling_embs(mesh1, mesh2, FM_12, params=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the coupling embeddings for each shape in order to compute p2p_21

    Parameters
    ----------
    mesh1 : pyFM.mesh.TriMesh
        First mesh
    mesh2 : pyFM.mesh.TriMesh
        Second mesh
    FM_12 : np.ndarray
        Functional map from mesh1 to mesh2
    params : OmegaConf DictConfig
        Parameters dictionary

    Returns
    -------
    emb_couple_21_1 : np.ndarray (n1, k1)
        Coupling embedding for mesh1
    emb_couple_21_2 : np.ndarray (n1, k1)
        Coupling embedding for mesh2
    """

    emb_couple_21 = emb.coupling_emb(FM_12, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k1), (n2, k1)

    emb_couple_21_1, emb_couple_21_2 = emb_couple_21
    return emb_couple_21_1, emb_couple_21_2

def get_spectral_orth_coupling_embs(mesh1, mesh2, FM_12, params=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the coupling embeddings for each shape in order to compute p2p_21

    Parameters
    ----------
    mesh1 : pyFM.mesh.TriMesh
        First mesh
    mesh2 : pyFM.mesh.TriMesh
        Second mesh
    FM_12 : np.ndarray
        Functional map from mesh1 to mesh2
    params : OmegaConf DictConfig
        Parameters dictionary

    Returns
    -------
    emb_couple_21_1 : np.ndarray (n1, k2)
        Coupling embedding for mesh1
    emb_couple_21_2 : np.ndarray (n2, k2)
        Coupling embedding for mesh2
    """
    emb_couple_21 = emb.orthogonal_emb(FM_12, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)


    emb_couple_21_1, emb_couple_21_2 = emb_couple_21
    return emb_couple_21_1, emb_couple_21_2

def get_spectral_bij_embs(mesh1, mesh2, FM_21, params=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the bijectivity embeddings for each shape in order to compute p2p_21

    Parameters
    ----------
    mesh1 : pyFM.mesh.TriMesh
        First mesh
    mesh2 : pyFM.mesh.TriMesh
        Second mesh
    FM_12 : np.ndarray
        Functional map from mesh1 to mesh2
    FM_21 : np.ndarray
        Functional map from mesh2 to mesh1
    params : OmegaConf DictConfig
        Parameters dictionary

    Returns
    -------
    emb_bij_21_1 : np.ndarray (n1, k2)
        Bijectivity embedding for mesh1 
    emb_bij_21_2 : np.ndarray (n2, k2)
        Bijectivity embedding for mesh2
    """


    emb_bij_21 = emb.bijectivity_emb(FM_21, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)
    
    emb_bij_21_1, emb_bij_21_2 = emb_bij_21  # (n1, k2), (n2, k2)
    return emb_bij_21_1, emb_bij_21_2

def get_spectral_conf_embs(mesh1, mesh2, FM_12, params=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the conformal embeddings for each shape in order to compute p2p_21

    Parameters
    ----------
    mesh1 : pyFM.mesh.TriMesh
        First mesh
    mesh2 : pyFM.mesh.TriMesh
        Second mesh
    FM_12 : np.ndarray (k2, k1)
        Functional map from mesh1 to mesh2

    Returns
    -------
    emb_conf_21_1 : np.ndarray (n1, k2)
        Conformal embedding for mesh1
    emb_conf_21_2 : np.ndarray (n2, k2)
        Conformal embedding for mesh2
    """
    emb_conf_21 = emb.conformal_emb(FM_12, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)
    emb_conf_21_1, emb_conf_21_2 = emb_conf_21  # (n1, k2), (n2, k2)
    return emb_conf_21_1, emb_conf_21_2

def get_spectral_descr_embs(mesh1, mesh2, FM_12, descr1, descr2, params=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the descriptor embeddings for each shape in order to compute p2p_21

    Parameters
    ----------
    mesh1 : pyFM.mesh.TriMesh
        First mesh
    mesh2 : pyFM.mesh.TriMesh
        Second mesh
    FM_12 : np.ndarray (k2, k1)
        Functional map from mesh1 to mesh2
    descr1 : np.ndarray (n1, d)
        Descriptors for mesh1
    descr2 : np.ndarray (n2, d)
        Descriptors for mesh2
    params : OmegaConf DictConfig
        Parameters dictionary

    Returns
    -------
    emb_descr_21_1 : np.ndarray (n1, d)
        Descriptor embedding for mesh1
    emb_descr_21_2 : np.ndarray (n2, d)
        Descriptor embedding for mesh2
    """

    k2, k1 = FM_12.shape

    f1 = mesh1.project(descr1, k=k1)
    f2 = mesh2.project(descr2, k=k2)

    emb_descr_21 = emb.descriptor_emb(FM_12, f1, f2, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, d), (n2, d)
    
    emb_descr_21_1, emb_descr_21_2 = emb_descr_21  # (n1, d), (n2, d)
    return emb_descr_21_1, emb_descr_21_2

def get_primal_coupling_emb(mesh1, mesh2, Y_21, params=None):
    emb_couple_21 = emb.coupling_emb_spatial(Y_21, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)

    return emb_couple_21


def get_primal_bij_emb(mesh1, mesh2, Y_21, Y_12, params=None):
    emb_bij_21 = emb.bijectivity_emb_spatial(Y_21, Y_12, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)

    return emb_bij_21