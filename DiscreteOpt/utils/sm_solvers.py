
import numpy as np
import scipy.sparse as sparse

from omegaconf import OmegaConf

from .smooth_methods import nicp, arap, dirichlet, rhm


def solve_Y21_arap_couple(mesh1, mesh2, p2p_21, params, landmarks=None):
    """
    Solve the ARAP problem with a couple term

    Parameters
    ----------
    mesh1 : Mesh
        First mesh
    mesh2 : Mesh
        Second mesh
    p2p_21 : numpy.ndarray (n2,) or (n2, 3)
        Correspondence between mesh2 and mesh1
    params : dict
        Smoothness Parameters, with parameters of the ARAP method in key 'method_params'
    landmarks : tuple of numpy.ndarray
        Landmarks for mesh1 and mesh2, in the form (lmks1, lmks2) where lmks1 and lmks2 are numpy.ndarray of shape (n_lmks,),
        representing the indices of the landmarks in mesh1 and mesh2 respectively

    Returns
    -------
    Y21 : numpy.ndarray (n2, 3)
        Estimated positions of the vertices of mesh1 on mesh2
    """

    smooth_weight = params['smooth_weight']
    couple_weight = params['couple_weight']
    
    nit = OmegaConf.select(params, 'method_params.nit', default=1)
    if landmarks is not None:
        lmks1, lmks2 = landmarks
        lmks = (lmks2, lmks1)
    else:
        lmks = None

    Y21 = arap.solve_arap_mesh_coupling(mesh1=mesh2, mesh2=mesh1, p2p_12=p2p_21, landmarks=lmks, couple_weight=couple_weight/smooth_weight, nit=nit, verbose=False)

    return Y21

def solve_Y21_nicp(mesh1, mesh2, p2p_21, params, landmarks=None):
    """
    Solve the NICP problem

    Parameters
    ----------
    mesh1 : Mesh
        First mesh
    mesh2 : Mesh
        Second mesh
    p2p_21 : numpy.ndarray (n2,) or (n2, 3)
        Correspondence between mesh2 and mesh1
    params : dict
        Smoothness Parameters, with parameters of the NICP method in key 'method_params', else default values are used
    landmarks : tuple of numpy.ndarray
        Landmarks for mesh1 and mesh2, in the form (lmks1, lmks2) where lmks1 and lmks2 are numpy.ndarray of shape (n_lmks,),
        representing the indices of the landmarks in mesh1 and mesh2 respectively

    Returns
    -------
    Y21 : numpy.ndarray (n2, 3)
        Estimated positions of the vertices of mesh1 on mesh2
    """
    
    smooth_weight = params['smooth_weight']
    couple_weight = params['couple_weight']

    skew_weight = OmegaConf.select(params, 'method_params.skew_weight', default=1e1)
    nit = OmegaConf.select(params, 'method_params.nit', default=1)
    use_cotan = OmegaConf.select(params, 'method_params.use_cotan', default=True)

    # Only if more than 1 iteration
    update_corr = OmegaConf.select(params, 'method_params.update_corr', default=False)
    n_jobs = OmegaConf.select(params, 'method_params.n_jobs', default=1)

    # Only if landmarks
    landmarks_weight = OmegaConf.select(params, 'method_params.landmarks_weight', default=1e2)
    if landmarks is not None:
        lmks1, lmks2 = landmarks
        lmks = (lmks2, lmks1)
    else:
        lmks = None


    Y21 = nicp.solve_nicp_mesh(mesh1=mesh2, mesh2=mesh1, p2p_12=p2p_21, skew_weight=skew_weight, stiffness_weight=smooth_weight/couple_weight, use_cotan=use_cotan,
                               landmarks=lmks, lmks_weight=landmarks_weight,
                               nit=nit, n_jobs=n_jobs, update_corr=update_corr, verbose=False)
    
    return Y21

def solve_Y21_dirichlet(mesh1, mesh2, p2p_21, params, landmarks=None):
    """
    Solve the exact problem

    Parameters
    ----------
    mesh1 : Mesh
        First mesh
    mesh2 : Mesh
        Second mesh
    p2p_21 : numpy.ndarray (n2,) or (n2, 3)
        Correspondence between mesh2 and mesh1
    params : dict
        Smoothness Parameters, with parameters of the Exact method in key 'method_params', else default values are used
    landmarks : tuple of numpy.ndarray
        Landmarks for mesh1 and mesh2, in the form (lmks1, lmks2) where lmks1 and lmks2 are numpy.ndarray of shape (n_lmks,),
        representing the indices of the landmarks in mesh1 and mesh2 respectively
        
    Returns
    -------
    Y21 : numpy.ndarray (n2, 3)
        Estimated positions of the vertices of mesh1 on mesh2
    """
    smooth_weight = params['smooth_weight']
    couple_weight = params['couple_weight']

    solver = OmegaConf.select(params, 'method_params.solver', default=None)   

    if landmarks is not None:
        lmks1, lmks2 = landmarks
        lmks = (lmks2, lmks1)
    else:
        lmks = None
    Y_21 = dirichlet.solve_dirichlet_coupling_mesh(mesh1=mesh2, mesh2=mesh1, p2p_12=p2p_21, couple_weight=couple_weight / smooth_weight, landmarks=lmks, solver=solver)

    return Y_21

def solve_Y21_rhm(mesh1, mesh2, p2p_21, p2p_12, params, landmarks=None):
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
    
    if landmarks is not None:
        lmks1, lmks2 = landmarks
        lmks = (lmks2, lmks1)
    else:
        lmks = None


    Y21 = rhm.solve_bijective_dirichlet_mesh(mesh1=mesh2, mesh2=mesh1, P12=P21, P21=P12,
                                             landmarks=lmks,
                                             couple_weight=w_couple/w_smooth, bijective_weight=w_bij/w_smooth)

    return Y21



def solve_Y21(mesh1, mesh2, p2p_21, p2p_12=None, params=None, landmarks=None):
    method = params["method"].lower()

    # if method == 'smooth_shells':
    #     Y_21 = solve_Y21_shells_proj(mesh1, mesh2, p2p_21, smooth_params, pb_or_def='pull-back')

    if method == 'arap':
        Y_21 = solve_Y21_arap_couple(mesh1, mesh2, p2p_21, params, landmarks=landmarks)

    elif method == 'nicp':
        Y_21 = solve_Y21_nicp(mesh1, mesh2, p2p_21, params, landmarks=landmarks)

    elif method == 'dirichlet':
        Y_21 = solve_Y21_dirichlet(mesh1, mesh2, p2p_21, params, landmarks=landmarks)

    elif method == 'rhm':
        Y_21 = solve_Y21_rhm(mesh1, mesh2, p2p_21, p2p_12, params, landmarks=landmarks)

    else:
        raise ValueError("Not Implemented")

    return Y_21






