# Parallely Sliced Barycenters

This folder contains the code computing parallely sliced Wasserstein barycenters (PSB).

Subfolder `src` contains the files implementing the main functions and classes for the computation of such barycenters. Those files are:

- `manifold.py`: contains the abstract class of manifolds and sliced manifolds, as well as classes implementing measures on manifolds.
- `sphere.py`: contains the class Sphere, for general computations on the sphere, and other inhereting classes as well as classes implementing measures on the sphere.
- `so3.py`: contains the class SO3, for general computations on SO3, and other inhereting classes.
- `barycenters.py`: contains the classes implementing parallely sliced Wasserstein barycenters (free and fixed-support), making use of manifolds' methods, as well as Wasserstein barycenters (free and fixed support) and Regularised Wasserstein barycentres (fixed support), calling the POT library, for comparison purpose.
- `ssb.py`: contains the class implementing the semicircular sliced Wasserstein barycenter (SSB) from [1], for comparison purpose.

The algorithm can be tested with `test_s2.py`, `test_sd.py` or `test_so3.py` using command `python test_....py -t [option]`. The most relevant options are

- for `test_s2.py`:
  - `convergence`: to see the evolution of loss of the PSB algorithm on the 2-sphere wrt the iterations (compared with SSB algorithm).
  - `time`: to see the time execution of the PSB algorihm on the 2-sphere according to the number of slices or projections (compared with SSB algorithm).
  - `shape`: to see the shape of the resulting barycenter for different input measures on the 2-sphere (compared with SSB algorithm).
  - `benchmark`: to see the shape of the PSB (fixed and free support), the SSB, the regularised and unregularised Wasserstein barycenter for a single input case on the 2-sphere.
- for `test_sd.py`:
  - `convergence`: to see the evolution of loss of the PSB algorithm on a sphere of arbitrary dimension wrt the iterations (compared with SSB algorithm).
  - `time`: to see the time execution of the PSB algorihm on spheres of arbitrary dimension according to the number of slices or projections (compared with SSB algorithm).
  - `shape`: to see the shape of the resulting barycenter for different input measures on the 3-sphere (compared with SSB algorithm).
- for `test_so3.py`:
  - `shape`: to see the shape of the resulting barycenter for different input measures on SO(3) (compared with the Wasserstein barycentre).

## References

[1] C. Bonet, P. Berg, N. Courty, F. Septier, L. Drumetz, and M. T. Pham, “Spherical Sliced-Wasserstein,” presented at the The Eleventh International Conference on Learning Representations, Sep. 2022. Available: <https://openreview.net/forum?id=jXQ0ipgMdU>