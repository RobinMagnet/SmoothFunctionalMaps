import numpy as np

def generate_klist(k_init, nit, step=1, k_max=None, log_space=False):
    if k_max is not None and step is not None:
        print("Warning: k_max and step are both defined, step will be ignored")

    if k_max is None:
        k_list = k_init + step * np.arange(1+nit, dtype=int)
    elif log_space:
        k_list = np.geomspace(k_init, k_max, num=1+nit, endpoint=True, dtype=int)
    else:
        k_list = np.linspace(k_init, k_max, num=1+nit, endpoint=True, dtype=int)

    # print(k_max)
    return k_list

def generate_weights_list(k_init, nit, step=1, k_max=None, log_space=False):
    if k_max is not None and step is not None:
        print("Warning: k_max and step are both defined, step will be ignored")

    if k_max is None:
        k_list = k_init + step * np.arange(1+nit, dtype=float)
    elif log_space:
        k_list = np.geomspace(k_init, k_max, num=1+nit, endpoint=True, dtype=float)
    else:
        k_list = np.linspace(k_init, k_max, num=1+nit, endpoint=True, dtype=float)

    # print(k_max)
    return k_list