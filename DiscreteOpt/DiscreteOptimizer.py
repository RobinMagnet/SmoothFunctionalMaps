
from tqdm.auto import tqdm

from omegaconf import OmegaConf, open_dict

from .utils import fm_solvers, spatial_solvers, opt_utils, params_utils, log_utils

class DiscreteOptimization:
    """
    Defines the Discrete optimization class, a generalization of ZoomOut, which alternatively solves for functional maps and pointwise maps
    """

    def __init__(self, mesh1, mesh2, p2p_21_init=None, p2p_12_init=None, descr1=None, descr2=None) -> None:
        """
        Initializes the DiscreteOptimization class
        
        Parameters
        ----------
        mesh1 : pyFM.mesh.TriMesh
            First mesh
        mesh2 : pyFM.mesh.TriMesh
            Second mesh
        p2p_21_init : np.ndarray
            Initial pointwise map from mesh2 to mesh1. Will be used to initialize the each optimization call (useful for testing multiple parameters)
        p2p_12_init : np.ndarray
            Initial pointwise map from mesh1 to mesh2. Will be used to initialize the each optimization call (useful for testing multiple parameters)
        """
        
        self.mesh1 = mesh1
        self.mesh2 = mesh2

        self.p2p_12 = None
        self.p2p_21 = None

        self.FM_12 = None
        self.FM_21 = None

        self.p2p_12_init = None
        self.p2p_21_init = None

        self.descr1 = descr1.copy() if descr1 is not None else None
        self.descr2 = descr2.copy() if descr2 is not None else None

        self.sp_params = None
        self.opt_params = None

        self._set_initial_p2p(p2p_21=p2p_21_init, p2p_12=p2p_12_init)
    
    def solve_FM_12(self, k: int):
        """
        Solves for the functional map from mesh1 to mesh2

        Parameters
        ----------
        k : int
            Number of eigenvectors to use
        """
        
        self.FM_12 = fm_solvers.solve_FM_12(mesh1=self.mesh1, mesh2=self.mesh2, k=k, p2p_21=self.p2p_21, p2p_12=self.p2p_12, params=self.sp_params)

    def solve_FM_21(self, k: int):
        """
        Solves for the functional map from mesh2 to mesh1

        Parameters
        ----------
        k : int
            Number of eigenvectors to use
        """
        self.FM_21 = fm_solvers.solve_FM_12(mesh1=self.mesh2, mesh2=self.mesh1, k=k, p2p_21=self.p2p_12, p2p_12=self.p2p_21, params=self.sp_params)

    def solve_p2p_12(self, precise=False, n_jobs=1):
        """
        Solves for the pointwise map from mesh1 to mesh2
        """
        self.p2p_12 = spatial_solvers.solve_p2p_21_spectral(mesh1=self.mesh2, mesh2=self.mesh1, FM_12=self.FM_21, FM_21=self.FM_12, descr1=self.descr2, descr2=self.descr1, params=self.sp_params, n_jobs=n_jobs, precise=precise)

    def solve_p2p_21(self, precise=False, n_jobs=1):
        """
        Solves for the pointwise map from mesh2 to mesh1
        """
        self.p2p_21 = spatial_solvers.solve_p2p_21_spectral(mesh1=self.mesh1, mesh2=self.mesh2, FM_12=self.FM_12, FM_21=self.FM_21, descr1=self.descr1, descr2=self.descr2, params=self.sp_params, n_jobs=n_jobs, precise=precise)

    @property
    def n_inner(self):
        return self.opt_params.get("n_inner", 1)
    
    def set_params(self, params=None, sp_params=None, opt_params=None):
        """
        Sets the parameters for the optimization

        Parameters
        ----------
        params : OmegaConf DictConfig or path to yaml file
            Parameters dictionary with sp_params and opt_params included
        sp_params : OmegaConf DictConfig or path to yaml file
            Spectral parameters dictionary
        opt_params : OmegaConf DictConfig or path to yaml file
            Optimization parameters dictionary
        """
        if params is not None:
            if not ((sp_params is None) and (opt_params is None)):
                raise ValueError("You must provide not sp_params and opt_params if you provide the full parameters")

            params = params_utils.load_params(params)
            self.sp_params = params.sp_params
            self.opt_params = params.opt_params

        elif (sp_params is None) or (opt_params is None):
            raise ValueError("You must provide either the full parameters or both sp_params and opt_params")
        
        else:
            self.sp_params = params_utils.load_params(sp_params)
            self.opt_params = params_utils.load_params(opt_params)

    def _preprocess_meta_info_params(self):
        """
        Preprocesses the parameters for the optimization
        """
        fm_type = fm_solvers._get_FM_type_from_params(self.sp_params)
        emb_types = spatial_solvers._get_spectral_embedding_types_from_params(self.sp_params)
        
        meta_info = {"meta_info": dict(FM_type=fm_type, emb_types=emb_types)}
        self.sp_params = OmegaConf.merge(self.sp_params, OmegaConf.create(meta_info))

        # OmegaConf.set_struct(self.sp_params, True)
        # with open_dict(self.sp_params):
        #     self.sp_params["meta_info"]["FM_type"] = fm_type
        #     self.sp_params["meta_info"]["emb_types"] = emb_types
        # print(self.sp_params["meta_info"])
        # raise ValueError("Not implemented")
        # ["FM_type"] = fm_type
        
    
    def print_emb_sizes(self):
        """
        Get the embedding sizes for the optimization
        """
        if self.sp_params is None:
            raise ValueError("You must set the parameters before getting the embedding sizes")
        
        emb_sizes = OmegaConf.select(self.sp_params, "meta_info.emb_sizes")
        if emb_sizes is None:
            self._preprocess_meta_info_params()

        emb_dim_K1, emb_dim_K2, emb_dim_p, emb_dim_bias = spatial_solvers._get_spectral_emb_dim_from_emb_types(self.sp_params.meta_info.emb_types)
        
        emb_size_str = log_utils.get_emb_size_str_base(emb_dim_K1=emb_dim_K1, emb_dim_K2=emb_dim_K2, emb_dim_p=emb_dim_p, emb_dim_bias=emb_dim_bias)
        print(emb_size_str)


    def _set_initial_p2p(self, p2p_21=None, p2p_12=None):
        """
        Initializes the pointwise maps from mesh2 to mesh1 and from mesh1 to mesh2
        """
        self.p2p_21_init = p2p_21.copy() if p2p_21 is not None else self.p2p_21_init
        self.p2p_12_init = p2p_12.copy() if p2p_12 is not None else self.p2p_12_init

    def _initialize_p2p(self, p2p_21=None, p2p_12=None):
        """
        Initializes the pointwise maps from mesh2 to mesh1 and from mesh1 to mesh2
        Use the initial values if not provided
        """

        self.p2p_21 = self.p2p_21_init.copy() if p2p_21 is None else p2p_21.copy()
        self.p2p_12 = self.p2p_12_init.copy() if p2p_12 is None else p2p_12.copy()

    def _initialize_FM(self, FM_12=None, FM_21=None):
        """
        Initializes the functional maps from mesh1 to mesh2 and from mesh2 to mesh1
        Use the initial values if not provided
        """
        self.FM_12 = FM_12.copy() if FM_12 is not None else None
        self.FM_21 = FM_21.copy() if FM_21 is not None else None
     
    def _initialize_descr(self, descr1=None, descr2=None):
        """
        Initializes the descriptors for the optimization
        """
        if descr1 is not None:
            self.descr1 = descr1.copy()
        if descr2 is not None:
            self.descr2 = descr2.copy()

    def solve_from_p2p(self, p2p_21=None, p2p_12=None, descr1=None, descr2=None, n_jobs=1, verbose=False):
        """
        Solves for the functional maps and pointwise maps
        """
        if self.sp_params is None or self.opt_params is None:
            raise ValueError("You must set the parameters before solving")
        
        self._initialize_p2p(p2p_21=p2p_21, p2p_12=p2p_12)
        self._initialize_descr(descr1=descr1, descr2=descr2)
        self.opt_params["n_jobs"] = n_jobs

        k_list = opt_utils.generate_klist(k_init=self.opt_params.k_init, nit=self.opt_params.nit, step=self.opt_params.step, k_max=self.opt_params.k_max, log_space=self.opt_params.log_space)

        pbar = tqdm(k_list) if verbose else k_list
        
        for it, k in enumerate(pbar):
            if verbose:
                pbar.set_description(f"Solving for k={k}")

            pbar2 = tqdm(range(self.n_inner), leave=False) if (verbose and self.n_inner > 1) else range(self.n_inner)
            for it_inner in pbar2:
                self.solve_FM_12(k)
                self.solve_FM_21(k)

                self.solve_p2p_12()

                
                self.solve_p2p_21()
        