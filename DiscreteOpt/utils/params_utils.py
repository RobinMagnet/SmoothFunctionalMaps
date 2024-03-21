import sys
import os

import omegaconf
from omegaconf import OmegaConf


def get_params_for_method(method: str) -> OmegaConf:
    """
    Get the parameters for the given method

    Parameters
    ----------
    method : str
        Method name

    Returns
    -------
    params : OmegaConf
        Parameters for the given method
    """
    # Seek the corresponding yaml file in ./params directory

    # Get the path to the params directory
    params_dir = os.path.join(os.path.dirname(__file__), "params")

    # Get the path to the yaml file
    yaml_file = os.path.join(params_dir, f"{method.lower()}.yml")

    if not os.path.isfile(yaml_file):
        raise ValueError(f"Method {method} not found in the params directory ({params_dir})")
    # Load the yaml file
    params = OmegaConf.load(yaml_file)

    return params

def generate_sp_params_from_template(method, **kwargs):
    """
    Generate the spatial parameters from the given template

    Parameters
    ----------
    method : str
        Method name
    kwargs : dict
        Additional parameters

    Returns
    -------
    sp_params : OmegaConf
        Spatial parameters
    """
    # Get the spatial parameters from the given method
    sp_params = get_params_for_method(method)["sp_params"]

    # check that the given kwargs are valid
    valid_keys = sp_params.keys()
    for key in kwargs:
        if key not in valid_keys:
            raise ValueError(f"Invalid key: {key}. Valid keys are: {valid_keys}")
            

    # Update the spatial parameters with the given kwargs
    sp_params = OmegaConf.merge(sp_params, OmegaConf.create(kwargs))

    return sp_params


def get_default_opt_params() -> OmegaConf:
    """
    Get the default optimization parameters

    Returns
    -------
    opt_params : OmegaConf
        Default optimization parameters
    """
    # Get the path to the yaml file
    opt_params = get_params_for_method("default")["opt_params"]

    return opt_params

def load_params(params):
    """
    Load the parameters from a given path or dict or OmegaConf.
    Return OmegaConf object

    Parameters
    ----------
    params : str or dict or OmegaConf
        Parameters

    Returns
    -------
    params : OmegaConf
        Parameters
    """

    if isinstance(params, str):
        if os.path.isfile(params):
            params = OmegaConf.load(params)
        else:
            params = get_params_for_method(params)
    elif isinstance(params, dict):
        params = OmegaConf.create(params)
    elif not isinstance(params, omegaconf.dictconfig.DictConfig):
        raise ValueError("params must be either a path to a yaml file, a methods name, a dictionary or an OmegaConf object")

    return params