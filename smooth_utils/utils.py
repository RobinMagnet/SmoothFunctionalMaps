import numpy as np
import scipy.sparse as sparse

import pyFM.spectral as spectral
import pyFM.refine
from pyFM.spectral import projection_utils as pju
from pyFM.spectral import nn_utils

from . import embeddings as emb
from . import smoothness_solvers as smsolve


def _base_sp_params():
    sp_params = {
        "use_bij": True,  # Use bijective energy

        "simple_energy": True,  # Only use coupling for FM computation (+ bij if here but ignore all other terms)
        "bij_only": True,  # Only coupling + bij in all energy.
        "simple_weights": True,  # Simpler weighting for FM computation (only if only bij+coupling)

        "orthogonal_FM": False,  # Outputs an orthogonal functional map, not recommended unless isometric shapes

        "couple_weight": 1,  # weight for coupling
        "bij_weight": 1,  # weight for bijective energy

        'global_weight': .25,  # Global reweight when using jointly with primal values

        'orth_coupling': False,  # Whether to use orthogonal coupling instead of standard coupling. Only if very isometric

        "only_coupling_for_nn": True,
        "normalize_terms": True
        }

    return sp_params


def _do_sp_param(method):
    if method == "zoomout":
        sp_params = {
            "use_bij": False,  # Use bijective energy

            "simple_energy": True,  # Only use coupling for FM computation (+ bij if here but ignore all other terms)
            "bij_only": False,  # Only coupling + bij in all energy.

            "simple_weights": False,  # Simpler weighting for FM computation (only if only bij+coupling)

            "orthogonal_FM": False,  # Outputs an orthogonal functional map, not recommended unless isometric shapes

            "couple_weight": 1,  # weight for coupling
            "bij_weight": 0,  # weight for bijective energy

            'orth_coupling': True,  # Whether to use orthogonal coupling instead of standard coupling. Only if very isometric

            "normalize_terms": False,
            "only_coupling_for_nn": True,
            }

    elif method == "bijective":
        sp_params = {
            "use_bij": True,  # Use bijective energy

            "simple_energy": True,  # Only use coupling for FM computation (+ bij if here but ignore all other terms)
            "bij_only": True,  # Only coupling + bij in all energy.

            "simple_weights": True,  # Simpler weighting for FM computation (only if only bij+coupling)

            "orthogonal_FM": False,  # Outputs an orthogonal functional map, not recommended unless isometric shapes

            "couple_weight": 1,  # weight for coupling
            "bij_weight": 1,  # weight for bijective energy
            "conf_weight": 1,

            'orth_coupling': True,  # Whether to use orthogonal coupling instead of standard coupling. Only if very isometric

            "normalize_terms": False,
            "only_coupling_for_nn": False,

            "conformal_energy": False
            }

    elif method == "all":
        sp_params = {
            "use_bij": True,  # Use bijective energy

            "simple_energy": True,  # Only use coupling for FM computation (+ bij if here but ignore all other terms)
            "bij_only": True,  # Only coupling + bij in all energy.

            "simple_weights": True,  # Simpler weighting for FM computation (only if only bij+coupling)

            "orthogonal_FM": False,  # Outputs an orthogonal functional map, not recommended unless isometric shapes

            "couple_weight": 1,  # weight for coupling
            "bij_weight": 1,  # weight for bijective energy
            "bij_weight": 1,  # weight for bijective energy

            'orth_coupling': True,  # Whether to use orthogonal coupling instead of standard coupling. Only if very isometric

            "normalize_terms": False,
            "only_coupling_for_nn": False,

            "conformal_energy": False
            }

    else:
        raise ValueError("unknown")

    return sp_params


def _base_sm_params(method):
    method = method.lower()
    if method == "rhm":
        sm_params = {
                    'smooth_weight': 1e0,
                    'couple_weight':  1e0,
                    'bij_weight': 1e4,

                    "global_weight": 1.5,  # Additional weight when merging with spectral
                    "method": method,

                    "only_coupling_for_nn": True,  # Might be worth to add but adds too much parameters
                    "normalize_terms": True,

                    }

    elif method in ["exact", "dirichlet"]:
        sm_params = {
                    'smooth_weight': 1e-3,
                    'couple_weight':  1e0,

                    "global_weight": 1.5,  # Additional weight when merging with spectral
                    "method": method,

                    "only_coupling_for_nn": True,  # Might be worth to add but adds too much parameters
                    "normalize_terms": True,
                    }

    elif method == "arap":
        sm_params = {
                    'smooth_weight': 1e1,
                    'couple_weight':  1e0,

                    "global_weight": 1.5,  # Additional weight when merging with spectral
                    "method": method,

                    "only_coupling_for_nn": True,  # Might be worth to add but adds too much parameters
                    "normalize_terms": True,
                    }

    elif method == "nicp":
        sm_params = {
                    'smooth_weight': 1e2,
                    'couple_weight':  1e0,

                    "global_weight": 1.5,  # Additional weight when merging with spectral
                    "method": method,

                    "only_coupling_for_nn": True,  # Might be worth to add but adds too much parameters
                    "normalize_terms": True,
                    }

    else:
        raise ValueError("Wrong method")

    return sm_params


def _base_opt_params(method):
    if method=="zoomout" or method=="bijective":
        opt_params = {
                      "k_init": 20,
                      "nit": 16,
                      "step": 5,
                      "n_inner": 1,
                      "k_max": None,
                      "log_space": False,

                      'sm_weight_range': [1e-1, 1e0]
                      }

    else:
        opt_params = {
                      "k_init": 10,
                      "nit": 10,
                      "step": 5,
                      "n_inner": 5,
                      "k_max": None,
                      "log_space": True,

                      'sm_weight_range': [1e-1, 1e0]
                      }
    return opt_params


def _scale_emb(embeddings, factor):
    emb1, emb2 = embeddings

    return factor * emb1, factor * emb2


def _get_FM_computation_type(params):
    FM_type = None

    if params['use_bij']:
        if params['simple_energy'] or params['bij_only']:
            if params['simple_weights']:
                FM_type = "bij_simple"

            else:
                FM_type = "bij_weighted"

        else:
            FM_type = "bij_complete"

    else:
        if params['simple_energy']:
            FM_type = "single_standard"

        else:
            FM_type = "single_complete"

    return FM_type


def _get_spectral_p2p_computation_type(params):
    if params["only_coupling_for_nn"]:
        emb_type = "coupling"

    elif params["use_bij"]:
        if params["conformal_energy"]:
            emb_type = "complete"

        else:
            emb_type = "coupling_bij"

    else:
        return "complete"

    return emb_type


def _get_primal_p2p_computation_type(params):
    if params["only_coupling_for_nn"]:
        emb_type = "coupling"

    else:
        emb_type = "coupling_bij"

    return emb_type


def solve_FM_12(mesh1, mesh2, k, p2p_21, p2p_12=None, params=None):
    FM_type = _get_FM_computation_type(params)

    # Standard unweighted bijective FM computation
    if FM_type == "bij_simple":
        sub1 = np.concatenate([np.arange(mesh1.n_vertices), p2p_21])
        sub2 = np.concatenate([p2p_12, np.arange(mesh2.n_vertices)])

        FM_12 = spectral.mesh_p2p_to_FM(np.arange(sub2.size), mesh1, mesh2,
                                        dims=int(k), subsample=(sub1,sub2))

    # Weighted bijective FM computation
    elif FM_type == "bij_weighted":
        couple_weight = params['couple_weight']
        bij_weight = params['bij_weight']

        ev1 = mesh1.eigenvectors[:, :k]
        ev2 = mesh2.eigenvectors[:, :k]
        ev1_pb = mesh1.eigenvectors[p2p_21, :k]
        ev2_pb = mesh2.eigenvectors[p2p_12, :k]

        A_mat = np.eye(k) + (bij_weight / couple_weight) * ev2_pb.T @ mesh1.A @ ev2_pb
        B_mat = ev2.T @ mesh2.A @ ev1_pb + (bij_weight/couple_weight) * ev2_pb.T @ mesh1.A @ ev1

        FM_12 = np.linalg.solve(A_mat, B_mat)

    # Bijective FM computation using all terms
    elif FM_type == "bij_complete":
        raise ValueError("Not Implemented Yet")

    # Standard pointwise to functional map conversion
    elif FM_type == "single_standard":
        FM_12 = spectral.mesh_p2p_to_FM(p2p_21, mesh1, mesh2, dims=k)

    # Pointwise to functional map conversion using all terms
    elif FM_type == "single_complete":
        pass

    else:
        raise ValueError("This should not happen")

    if params['orthogonal_FM']:
        FM_12 = pyFM.refine.mesh_icp_refine(mesh1, mesh2, FM_12, nit=1, use_adj=False,
                                            n_jobs=1, verbose=False)

    return FM_12


def get_spectral_coupling_emb(mesh1, mesh2, FM_12, params=None):
    if params["orth_coupling"]:
        emb_couple_21 = emb.orthogonal_emb(FM_12, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)

    else:
        emb_couple_21 = emb.coupling_emb(FM_12, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)

    return emb_couple_21


def get_spectral_bij_emb(mesh1, mesh2, FM_12, FM_21, params=None):
    emb_bij_21 = emb.bijectivity_emb(FM_12, FM_21, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)

    return emb_bij_21


def get_spectral_conf_emb(mesh1, mesh2, FM_12, FM_21, params=None):
    emb_conf_21 = emb.conformal_emb(FM_12, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)

    return emb_conf_21


def get_spectral_descr_emb(mesh1, mesh2, FM_12, params=None):

    ind1 = params["ind1"]
    ind2 = ind1 % 2 + 1

    f1 = params[f"descr{ind1}"]
    f2 = params[f"descr{ind2}"]

    k2, k1 = FM_12.shape

    descr1 = mesh1.project(f1, k=k1)
    descr2 = mesh2.project(f2, k=k2)

    emb_descr_21 = emb.conformal_emb(FM_12, descr1, descr2, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)

    return emb_descr_21


def get_spectral_embeddings_21(mesh1, mesh2, FM_12, FM_21=None, params=None):

    emb_type = _get_spectral_p2p_computation_type(params)

    emb_couple = get_spectral_coupling_emb(mesh1, mesh2, FM_12, params=params)
    emb_couple = _scale_emb(emb_couple, np.sqrt(params["couple_weight"])) if params["couple_weight"] != 1 else emb_couple

    # print("Spectr reweight", np.sqrt())

    if emb_type == "coupling":
        if "global_reweight" in params.keys() and params["global_reweight"] != 1:
            emb_couple = _scale_emb(emb_couple, np.sqrt(params["global_reweight"]))
        return emb_couple

    emb_total1 = [emb_couple[0]]
    emb_total2 = [emb_couple[1]]

    if emb_type == "coupling_bij" or params["bij_weight"] > 0:
        emb_bij = get_spectral_bij_emb(mesh1, mesh2, FM_12, FM_21, params=params)
        emb_bij = _scale_emb(emb_bij, np.sqrt(params["bij_weight"]))

        emb_total1.append(emb_bij[0])
        emb_total2.append(emb_bij[1])

    if emb_type == "complete":
        if "conf_weight" in params.keys() and params["conf_weight"] > 0:
            emb_conf = get_spectral_conf_emb(mesh1, mesh2, FM_12, params=params)
            emb_conf = _scale_emb(emb_conf, np.sqrt(params["conf_weight"]))

            emb_total1.append(emb_conf[0])
            emb_total2.append(emb_conf[1])

        if "descr_weight" in params.keys() and params["descr_weight"] > 0:
            emb_descr = get_spectral_descr_emb(mesh1, mesh2, FM_12, params=params)
            emb_descr = _scale_emb(emb_descr, np.sqrt(params["descr_weight"]))

            emb_total1.append(emb_descr[0])
            emb_total2.append(emb_descr[1])

    emb1 = np.concatenate(emb_total1, axis=1)
    emb2 = np.concatenate(emb_total2, axis=1)

    if "global_reweight" in params.keys():
        emb1 *= params["global_reweight"]
        emb2 *= params["global_reweight"]

    return emb1, emb2


def get_primal_coupling_emb(mesh1, mesh2, Y_21, params=None):
    emb_couple_21 = emb.coupling_emb_spatial(Y_21, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)

    return emb_couple_21


def get_primal_bij_emb(mesh1, mesh2, Y_21, Y_12, params=None):
    emb_bij_21 = emb.bijectivity_emb_spatial(Y_21, Y_12, mesh1, mesh2, normalize=params["normalize_terms"])  # (n1, k2), (n2, k2)

    return emb_bij_21


def get_primal_embeddings_21(mesh1, mesh2, Y_21, Y_12=None, params=None):
    emb_type = _get_primal_p2p_computation_type(params)

    emb_couple = get_primal_coupling_emb(mesh1, mesh2, Y_21, params=params)
    emb_couple = _scale_emb(emb_couple, np.sqrt(params["couple_weight"]))

    #print("Primal reweight", np.sqrt(params["global_reweight"])*np.sqrt(params["couple_weight"]) / np.sqrt(mesh2.l2_sqnorm(Y_21).sum()))

    if emb_type == "coupling":
        if "global_reweight" in params.keys():
            emb_couple = _scale_emb(emb_couple, np.sqrt(params["global_reweight"]))
        return emb_couple

    emb_total1 = [emb_couple[0]]
    emb_total2 = [emb_couple[1]]

    if emb_type == "coupling_bij":
        emb_bij = get_primal_bij_emb(mesh1, mesh2, Y_21, Y_12, params=params)
        emb_bij = _scale_emb(emb_bij, np.sqrt(params["bij_weight"]))

        emb_total1.append(emb_bij[0])
        emb_total2.append(emb_bij[1])

    emb1 = np.concatenate(emb_total1, axis=1)
    emb2 = np.concatenate(emb_total2, axis=1)

    if "global_reweight" in params.keys():
        emb1 *= params["global_reweight"]
        emb2 *= params["global_reweight"]

    return emb1, emb2


def solve_p2p_21_with_primal(mesh1, mesh2, FM_12, Y_21, FM_21=None, Y_12=None, params_sp=None, params_sm=None, n_jobs=1, precise=False):
    emb1_sp, emb2_sp = get_spectral_embeddings_21(mesh1, mesh2, FM_12, FM_21=FM_21, params=params_sp)
    emb1_pr, emb2_pr = get_primal_embeddings_21(mesh1, mesh2, Y_21, Y_12=None, params=params_sm)

    # emb1 = np.concatenate([emb1_sp, emb1_pr], axis=1)
    # emb2 = np.concatenate([emb2_sp, emb2_pr], axis=1)
    emb1 = np.concatenate([emb1_pr, emb1_sp], axis=1)
    emb2 = np.concatenate([emb2_pr, emb2_sp], axis=1)

    # return emb1, emb2

    if precise:
        p2p_21 = pju.project_pc_to_triangles(emb1, mesh1.facelist, emb2, n_jobs=n_jobs)
    else:
        p2p_21 = nn_utils.knn_query(emb1, emb2, n_jobs=n_jobs)

    return p2p_21


def solve_p2p_21_spectral(mesh1, mesh2, FM_12, FM_21=None, params=None, n_jobs=1, precise=False):
    emb1, emb2 = get_spectral_embeddings_21(mesh1, mesh2, FM_12, FM_21=FM_21, params=params)

    if precise:
        p2p_21 = pju.project_pc_to_triangles(emb1, mesh1.facelist, emb2, n_jobs=n_jobs)
    else:
        p2p_21 = nn_utils.knn_query(emb1, emb2, n_jobs=n_jobs)

    return p2p_21


def solve_Y21(mesh1, mesh2, p2p_21, p2p_12=None, params=None):
    method = params["method"].lower()

    # if method == 'smooth_shells':
    #     Y_21 = solve_Y21_shells_proj(mesh1, mesh2, p2p_21, smooth_params, pb_or_def='pull-back')

    if method == 'arap':
        Y_21 = smsolve.solve_Y21_arap_couple(mesh1, mesh2, p2p_21, params)

    elif method == 'nicp':
        Y_21 = smsolve.solve_Y21_nicp(mesh1, mesh2, p2p_21, params)

    elif method == 'exact':
        Y_21 = smsolve.solve_Y21_exact(mesh1, mesh2, p2p_21, params)

    elif method == 'rhm':
        Y_21 = smsolve.solve_Y21_exact_bij(mesh1, mesh2, p2p_21, p2p_12, params)

    else:
        raise ValueError("Not Implemented")

    return Y_21


def generate_klist(k_init, nit, step, k_max, log_space=False):
    if k_max is None:
        k_list = k_init + step * np.arange(1+nit, dtype=int)
    elif log_space:
        k_list = np.geomspace(k_init, k_max, num=nit, endpoint=True, dtype=int)
    else:
        k_list = np.linspace(k_init, 1+k_max, num=nit, endpoint=True, dtype=int)

    # print(k_max)
    return k_list


def generate_solver(mesh1, smooth_weight, couple_weight):

    system_matrix = couple_weight / (smooth_weight * mesh1.area**2) * mesh1.A
    system_matrix += 1 / mesh1.area * mesh1.W

    solver = sparse.linalg.factorized(system_matrix.tocsc())

    return solver
