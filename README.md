# Discrete Optimization and Smooth Discrete Optimization

This repository implements the following two papers:
  - J. Ren, S. Melzi, P. Wonka, and M. Ovsjanikov. *[Discrete Optimization for Shape Matching](https://www.lix.polytechnique.fr/~maks/papers/SGP21_DiscMapOpt.pdf)*. Computer Graphics Forum, 40(5), 2021.
  - R. Magnet, J. Ren, O. Sorkine-Hornung, and M. Ovsjanikov. *[Smooth NonRigid Shape Matching via Effective Dirichlet Energy Optimization](https://www.lix.polytechnique.fr/Labo/Robin.Magnet/3DV2022_smooth_corres/smooth_corres_main.pdf)*. In 2022 International Conference on 3D Vision (3DV).

The repository is now in version 1.0.0 where all features from both papers are implemened.

## Dependencies and install

If you have [pyfmaps](https://github.com/RobinMagnet/pyFM) installed, you can remove the `pyFM` repo here.
Else clone using
```
git clone --recurse-submodules https://github.com/RobinMagnet/smoothFM.git
```

## Code
Run the `Example Notebook.ipynb' for instructions on how to use the algorithms.

The code relies on the use of yaml config files, see in [this directory](./DiscreteOpt/utils/params/).

## Documentation
Better Documentation is on its way. This repo will eventually be merged in the [pyFM](https://github.com/RobinMagnet/pyFM) repo.
