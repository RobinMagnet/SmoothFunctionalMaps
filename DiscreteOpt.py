import copy
import numpy as np
from smooth_utils import utils

from tqdm.auto import tqdm


class DiscreteOptimization():
    """
    This class implements the Discrete Optimization Algorithm with several option.
    """

    def __init__(self, mesh1, mesh2, p2p_12=None, p2p_21=None):
        """
        Intialize the class

        Parameters
        ----------------------
        mesh1 : pyFM.TriMesh - First mesh
        mesh2 : pyFM.TriMesh - Second mesh
        p2p_12 : (n1,) array - Optional. Initial map from mesh1 to mesh2
        p2p_21 : (n2,) array - Optional. Initial map from mesh2 to mesh1
        """

        self.mesh1 = copy.deepcopy(mesh1)
        self.mesh2 = copy.deepcopy(mesh2)

        self.opt_params = None
        self.sp_params = None

        self.p2p_12 = None
        self.p2p_21 = None

        self.FM_12 = None
        self.FM_21 = None

        self.p2p_12_init = None
        self.p2p_21_init = None

        self.set_init(p2p_12=p2p_12, p2p_21=p2p_21)

    def _set_params(self, method):
        """
        This uses preset parameters for discrete optimization

        Parameters:
        -------------------
        method : str - Name of the preset, either "bijective" for bijective zoomout or "zoomout" for standard zoomout
        """
        self.sp_params = utils._do_sp_param(method)
        self.opt_params = utils._base_opt_params(method)

    def solve_FM_12(self, k):
        """
        Compute the functional map going from mesh1 to mesh2

        Parameters:
        -------------------
        k : int or tuple - shape of the functional map. If an int the shape will be (k,k)
        """
        self.FM_12 = utils.solve_FM_12(self.mesh1, self.mesh2, k, self.p2p_21, p2p_12=self.p2p_12, params=self.sp_params)

    def solve_FM_21(self, k):
        """
        Compute the functional map going from mesh2 to mesh1

        Parameters:
        -------------------
        k : int or tuple - shape of the functional map. If an int the shape will be (k,k)
        """
        self.FM_21 = utils.solve_FM_12(self.mesh2, self.mesh1, k, self.p2p_12, p2p_12=self.p2p_21, params=self.sp_params)

    def solve_p2p_12(self, n_jobs=1, precise=False):
        """
        Compute the pointwise map going from mesh1 to mesh2

        Parameters:
        -------------------
        n_jobs  : int or -1 - number of parallel jobs for nearest neighbors
        precise : produces vertex to point (in barycentric coordinates) instead of vertex to vertex
        """
        self.sp_params["ind1"] = 2
        self.p2p_12 = utils.solve_p2p_21_spectral(self.mesh2, self.mesh1, self.FM_21, FM_21=self.FM_12,
                                                  params=self.sp_params, n_jobs=n_jobs, precise=precise)

    def solve_p2p_21(self, n_jobs=1, precise=False):
        """
        Compute the pointwise map going from mesh2 to mesh1

        Parameters:
        -------------------
        n_jobs  : int or -1 - number of parallel jobs for nearest neighbors
        precise : produces vertex to point (in barycentric coordinates) instead of vertex to vertex
        """
        self.sp_params["ind1"] = 1
        self.p2p_21 = utils.solve_p2p_21_spectral(self.mesh1, self.mesh2, self.FM_12, FM_21=self.FM_21,
                                                  params=self.sp_params, n_jobs=n_jobs, precise=precise)

    def generate_klist(self):
        """
        Generates the list of sizes the functional maps will take from the parameters

        Output
        --------------------
        k_list : list of functional map size
        """
        k_init = self.opt_params["k_init"]
        nit = self.opt_params["nit"]
        step = self.opt_params["step"]
        k_max = self.opt_params["k_max"]
        log_space = self.opt_params["log_space"]

        return utils.generate_klist(k_init, nit, step, k_max, log_space=log_space)

    def set_init(self, p2p_12=None, p2p_21=None):
        """
        Sets the initial maps

        Parameters:
        --------------------
        p2p_12 : (n1,) array - initial map from mesh1 to mesh2
        p2p_21 : (n2,) array - initial map from mesh2 to mesh1
        """
        if p2p_12 is not None:
            self.p2p_12_init = p2p_12.copy()

        if p2p_21 is not None:
            self.p2p_21_init = p2p_21.copy()

    def _initialize(self, p2p_21=None, p2p_12=None):
        """
        Initialize the algorithm using preset or given intial maps.

        Parameters:
        --------------------
        p2p_12 : (n1,) array - initial map from mesh1 to mesh2
        p2p_21 : (n2,) array - initial map from mesh2 to mesh1
        """
        if p2p_21 is not None:
            self.p2p_21 = p2p_21.copy()
        elif self.p2p_21_init is not None:
            self.p2p_21 = self.p2p_21_init.copy()
        else:
            raise ValueError("NO INITIAL 2->1 MAP")

        if p2p_12 is not None:
            self.p2p_12 = p2p_12.copy()
        elif self.p2p_12_init is not None:
            self.p2p_12 = self.p2p_12_init.copy()
        else:
            raise ValueError("NO INITIAL 1->2 MAP")

        self.FM_12 = None
        self.FM_21 = None

    def fit(self, p2p_21=None, p2p_12=None, n_jobs=1, verbose=False):
        """
        Fit the algorithm using the parameters. Initialization can be given here.

        Parameters:
        --------------------
        p2p_12 : (n1,) array - initial map from mesh1 to mesh2
        p2p_21 : (n2,) array - initial map from mesh2 to mesh1
        n_jobs  : int or -1 - number of parallel jobs for nearest neighbors
        precise : produces vertex to point (in barycentric coordinates) maps instead of vertex to vertex
        """

        self._initialize(p2p_21=p2p_21, p2p_12=p2p_12)

        k_list = self.generate_klist()
        if verbose:
            k_list = tqdm(k_list)

        for it, k_curr in enumerate(k_list):

            for it_inner in range(self.opt_params["n_inner"]):

                self.solve_FM_12(k_curr)
                self.solve_FM_21(k_curr)

                self.solve_p2p_21(n_jobs=n_jobs)
                self.solve_p2p_12(n_jobs=n_jobs)


class SmoothDiscreteOptimization2(DiscreteOptimization):
    """
    This class extends the DiscreteOptimization class with primal energies
    """

    def __init__(self, mesh1, mesh2, p2p_12=None, p2p_21=None):
        """
        Intialize the class

        Parameters
        ----------------------
        mesh1 : pyFM.TriMesh - First mesh
        mesh2 : pyFM.TriMesh - Second mesh
        p2p_12 : (n1,) array - Optional. Initial map from mesh1 to mesh2
        p2p_21 : (n2,) array - Optional. Initial map from mesh2 to mesh1
        """

        super().__init__(mesh1, mesh2, p2p_12=p2p_12, p2p_21=p2p_21)

        self.sm_params = None

        self.Y_12 = None
        self.Y_21 = None

        self.smooth_method = None

    def _set_params(self, method):
        """
        This uses preset parameters for the algorithm

        Parameters:
        -------------------
        method : str - Name of the preset, either "dirichlet" for dirichlet energy only,
                 "rhm" for RHM energy, "arap" for ARAP energy and "nicp" for non rigid ICP energy.
        """
        self.sp_params = utils._base_sp_params()
        self.opt_params = utils._base_opt_params(method)
        self.sm_params = utils._base_sm_params(method)

    def solve_FM_12(self, k):
        """
        Compute the functional map going from mesh1 to mesh2

        Parameters:
        -------------------
        k : int or tuple - shape of the functional map. If an int the shape will be (k,k)
        """
        self.FM_12 = utils.solve_FM_12(self.mesh1, self.mesh2, k, self.p2p_21, p2p_12=self.p2p_12, params=self.sp_params)

    def solve_FM_21(self, k):
        """
        Compute the functional map going from mesh2 to mesh1

        Parameters:
        -------------------
        k : int or tuple - shape of the functional map. If an int the shape will be (k,k)
        """
        self.FM_21 = utils.solve_FM_12(self.mesh2, self.mesh1, k, self.p2p_12, p2p_12=self.p2p_21, params=self.sp_params)

    def solve_Y_12(self):
        """
        Compute the estimated pull-back coordinates using the map from mesh1 to mesh2
        """
        self.sm_params['solver_ind'] = 1
        self.Y_12 = utils.solve_Y21(self.mesh2, self.mesh1, self.p2p_12, p2p_12=self.p2p_21, params=self.sm_params)

    def solve_Y_21(self):
        """
        Compute the estimated pull-back coordinates using the map from mesh2 to mesh1
        """
        self.sm_params['solver_ind'] = 2
        self.Y_21 = utils.solve_Y21(self.mesh1, self.mesh2, self.p2p_21, p2p_12=self.p2p_12, params=self.sm_params)

    def solve_p2p_12(self, n_jobs=1, precise=False):
        """
        Compute the pointwise map going from mesh1 to mesh2

        Parameters:
        -------------------
        n_jobs  : int or -1 - number of parallel jobs for nearest neighbors
        precise : produces vertex to point (in barycentric coordinates) instead of vertex to vertex
        """
        self.p2p_12 = utils.solve_p2p_21_with_primal(self.mesh2, self.mesh1, self.FM_21, self.Y_12, FM_21=self.FM_12, Y_12=self.Y_21,
                                                     params_sp=self.sp_params, params_sm=self.sm_params, n_jobs=n_jobs, precise=precise)

    def solve_p2p_21(self, n_jobs=1, precise=False):
        """
        Compute the pointwise map going from mesh2 to mesh1

        Parameters:
        -------------------
        n_jobs  : int or -1 - number of parallel jobs for nearest neighbors
        precise : produces vertex to point (in barycentric coordinates) instead of vertex to vertex
        """
        self.p2p_21 = utils.solve_p2p_21_with_primal(self.mesh1, self.mesh2, self.FM_12, self.Y_21, FM_21=self.FM_21, Y_12=self.Y_12,
                                                     params_sp=self.sp_params, params_sm=self.sm_params, n_jobs=n_jobs, precise=precise)

    def _prefactor_system(self):
        """
        In the case of the "dirichlet" method, pre-factorize the sparse linear system
        """
        self.sm_params['solver1'] = utils.generate_solver(self.mesh1, self.sm_params['smooth_weight'], self.sm_params['couple_weight'])
        self.sm_params['solver2'] = utils.generate_solver(self.mesh2, self.sm_params['smooth_weight'], self.sm_params['couple_weight'])

    def generate_smooth_reweight_list(self):
        """
        Generates logarithmic weighting for increasing weights of primal energy.
        Depends on the parameters.

        Output
        --------------------
        weights : (nit,) - list of weights
        """
        w_min, w_max = self.opt_params['sm_weight_range']
        nit = self.opt_params["nit"]
        smooth_reweight_list = np.geomspace(w_min, w_max, 1+nit)

        return smooth_reweight_list

    def _initialize(self, p2p_21=None, p2p_12=None):
        """
        Initialize the algorithm using preset or given intial maps.

        Parameters:
        --------------------
        p2p_12 : (n1,) array - initial map from mesh1 to mesh2
        p2p_21 : (n2,) array - initial map from mesh2 to mesh1
        """
        # Use the function from the DiscreteOptimization class
        super()._initialize(p2p_21=None, p2p_12=None)

        self.Y_12 = None
        self.Y_21 = None

        if self.sm_params["method"] in ['exact', "dirichlet"]:
            self._prefactor_system()

    def fit(self, p2p_21=None, p2p_12=None, n_jobs=1, verbose=False):

        self._initialize(p2p_21=p2p_21, p2p_12=p2p_12)

        k_list = self.generate_klist()
        if verbose:
            k_list = tqdm(k_list)

        smooth_weight_list = self.generate_smooth_reweight_list()

        for it, k_curr in enumerate(k_list):

            self.sp_params['global_reweight'] = self.sp_params["global_weight"]
            self.sm_params['global_reweight'] = smooth_weight_list[it] * self.sm_params["global_weight"]

            for it_inner in range(self.opt_params["n_inner"]):

                self.solve_FM_12(k_curr)
                self.solve_FM_21(k_curr)

                self.solve_Y_21()
                self.solve_Y_12()

                self.solve_p2p_21(n_jobs=n_jobs)
                self.solve_p2p_12(n_jobs=n_jobs)
