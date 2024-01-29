import torch
import numpy as np
from math import ceil, sqrt
from scipy.stats import linregress

from src.manifold import SlicedManifold
import ot
from utils import unsqueeze, tqdm

import matplotlib.pyplot as plt



def sweep_matrix_index(NX, NY):
    """Computes the transport plan between two sets of points (i.e. two discrete measures expressed as \
        equiweighted sums of Dirac measures) with potentially different sizes, and its support.

    Args:
        NX (int): Number of support points of the first distribution.
        NY (int): Number of support points of the second distribution.

    Returns:
        sweep_reduced (Tensor): Shape (NX, n) with n = ceil(NY/NX) + 1. Transport plan on its support.
        sweep_index (Tensor): Shape (NX, n). Support of the transport plan (indices in range(0, NY)).
    """
    n = int(ceil(NY/NX)) + 1
    rx = torch.arange(NX)
    a, b = rx*NY/NX, (rx+1)*NY/NX
    deb = torch.min(torch.floor(a).long(), torch.tensor(NY - n))
    sweep_index = deb[:, None] + torch.arange(n)

    sweep_reduced = torch.clip(b[:, None] - sweep_index, 0, 1) - torch.clip(a[:, None] - sweep_index, 0, 1)
    sweep_reduced /= NY
    return sweep_reduced, sweep_index

def tau_generator(tau_init, a_tau=0.1, pow_tau=0):
    k=0
    while True:
        yield tau_init/(a_tau*k + 1)**pow_tau
        k += 1

class LSWBarycenter():
    """Abstract class computing the free-support Sliced Wasserstein Barycentres. It is aimed at representing \
    a specific operator on the manifold (which project the measures on the real line; and which are here \
    abstract methods) and implements, based on that, the stochastic gradient descent loop giving the sliced \
    Wasserstein barycentre. #!

    It is designed to be used in cooperative inheritance, along with a specific manifold, or directly
    inherited to define a generic operator (working on several manifold).
    """
    def __init__(self, sman:SlicedManifold):
        """
        Args:
            sman (SlicedManifold): sliced manifold
        """
        self.sman = sman

        self.measures = None
        self.lambdas = None
        self.barycenter = None
        self.frechet_quantity = None

        self.L_loss = []
        self.L_step = []
        self.L = []
        self.number_save = 20
        self.step_save = None

        self.L_stop = []
    
    @property
    def it_save_range(self):
        return torch.arange(len(self.L))*self.step_save
    


    def align_sliced_diff(self, psis, X, Y):
        """Computes the differences between the aligned projections of points in X and points in Y.
        Aligned means we ordonate the points before computing the differences. X and Y can contain several
        sets of points, in which case the operation is done for every pair of sets.

        Args:
            psis (Tensor): Shape (n, *ms). References on the manifold, with which we define the projections (slicing operation)
            X (Tensor): Shape (*Xs, N, *ms). Points (or set(s) of points) on the manifold.
            Y (Tensor): Shape (*Ys, N, *ms). Points (of set(s) of points) on the manifold.

        Returns:
            diff (Tensor): Shape (*Xs, *Ys, N, n). Aligned differences of the projections.
        """
        X_p = self.sman.operator(psis, X) #(*Xs, N, n)
        Y_p = self.sman.operator(psis, Y) #(*Ys, N, n)
        nXs = len(X_p.shape) - 2
        nYs = len(Y_p.shape) - 2

        # Comments below suppose Xs=Ys=(). When it is not the case, the operation is done for every entry.
        sX = torch.argsort(X_p, dim= -2)    # each column q contains \sigma_{X,\psi_q}      #(*Xs, N, n)
        sX_inv = torch.argsort(sX, dim= -2) # each column q contains \sigma^{-1}_{X,\psi_q} #(*Xs, N, n)
        sY = torch.argsort(Y_p, dim= -2)    # each column q contains \sigma_{Y, \psi_q}     #(*Ys, N, n)

        sY = unsqueeze(sY, 0, nXs)           # (1, ..., 1,   *Ys    , N, n)
        sX_inv = unsqueeze(sX_inv, nXs, nYs) # (  *Xs    , 1, ..., 1, N, n)
        sY_sXinv = torch.take_along_dim(sY, sX_inv, dim= -2) 
        # (*Xs, *Ys, N, n). Permute the element of each sY[:,q], the same way we would do to sort sX[:,q].
        # said differently, each column q of sY_sXinv contains \sigma_{Y, \psi_q} \circ \sigma^{-1}_{X, \psi_q}

        Y_p = unsqueeze(Y_p, 0  , nXs) # (1, ..., 1,   *Ys    , N, n)
        X_p = unsqueeze(X_p, nXs, nYs) # (  *Xs    , 1, ..., 1, N, n)
        diff = X_p - torch.take_along_dim(Y_p, sY_sXinv, dim= -2) 
        # (*Xs, *Ys, N, n). e.g. diff[k, q] = <X_k, psi_q> - <Y_{\sigma_{Y, \psi_q}( \sigma^{-1}_{X, \psi_q}(k) ),  \psi_q>

        return diff

    def sliced_dist_square_same(self, psis, X, Y):
        """Computes the squared sliced Wasserstein distance between the emprical measures represented by the points
        in X and Y, for the different slices psis. X and Y can contain different sets of points, each one representing
        a measure, in which case, the distances are computed for every pair of sets.

        Args:
            psis (Tensor): Shape (n, *ms). Points on the manifold representing the different slices.
            X (Tensor): Shape (*Xs, N, *ms). Points (or sets of points) representing empirical measure(s).
            Y (Tensor): Shape (*Ys, N, *ms). Points (or sets of points) representing empirical measure(s).

        Returns:
            dist (Tensor): Shape (*Xs, *Ys). Squared sliced distance between the two measures (or every pairs of measures).
        """
        diff = self.align_sliced_diff(psis, X, Y) # (*Xs, *Ys, N, n)
        return torch.mean(diff**2, dim=(-2, -1))  # (*Xs, *Ys)


    def functional_grad(self, psis, X, Y, lambdas, debug=False, it=None, **kwargs):
        """Compute the gradient of the Frechet functional (the one we have to minimise to get the Frechet barycentre) using the 
        sliced Wasserstein distance.

        Args:
            psis (Tensor): Shape (n, *ms). Points on the manifold representing the different slices.
            X (Tensor): Shape (N, *ms). Points representing the empirical measure with which we sequentially approximate the barycentre.
            Y (Tensor): Shape (M, N, *ms). Sets of points representing the empirical measures of which we want to compute the barycentre.
            lambdas (Tensor): Shape (M,). Weights of the different measure in the barycentre.

        Returns:
            gradient (Tensor): Shape (N, *ms). gradient of the Frechet functional in X.
            loss (float | None): The value of the Frechet functional.
        """
        n = psis.shape[0]
        N = X.shape[0]
        diff = self.align_sliced_diff(psis, X, Y) # (M, N, n)
        
        square_distances = torch.mean(diff**2, dim=(-2, -1)) # (M,)
        loss = lambdas.dot(square_distances) 

        gf = self.sman.operator_gradient_factor(psis, X, **kwargs) # (N, n)
        gradient_per_Yi = 2/(N*n) * torch.tensordot(diff * gf, psis, dims=([2], [0])) # (M, N, *ms)
        gradient = torch.tensordot(lambdas, gradient_per_Yi, dims=1) # (N, *ms)

        return gradient, loss
    
    # General case: different numbers of points in the measures.
    def align_sliced_diff_gnrl(self, psis, X, Y):
        """Computes the differences between the projections of points in X and points in Y which are linked by 
        the optimal transport plan. As we know that the transport plan's corresponding matrix is sparse, the 
        method computes only those pairs which are "connected" by this optimal transport plan (less than Nx+Ny+1)
        (plus few others, for convenience).

        Args:
            psis (Tensor): Shape (n, *ms). References on the manifold, with which we define the projections (slicing operation)
            X (Tensor): Shape (*Xs, Nx, *ms). Points (or set(s) of points) on the manifold.
            Y (Tensor): Shape (*Ys, Ny, *ms). Points (of set(s) of points) on the manifold.

        Returns:
            diff (Tensor): Shape (*Xs, *Ys, Nx, k, n). Aligned differences of the projections. k=ceil(Ny/Nx) + 1
            sweep_red (Tensor): Shape (Nx, k, 1). Weights associated to every pair. Sums to 1.
        """
        Xp = self.sman.operator(psis, X) # (*Xs, Nx, n) # X projected
        Yp = self.sman.operator(psis, Y) # (*Ys, Ny, n)
        nXs = len(Xp.shape) - 2
        nYs = len(Yp.shape) - 2
        Nx = Xp.shape[-2]
        Ny = Yp.shape[-2]
        
        # We keep only the useful Y_j
        sweep_red, sweep_idx = sweep_matrix_index(Nx, Ny) # (Nx, k) where k = ceil(Ny/Nx) + 1
        Xp_sort, sX = torch.sort(Xp, dim= -2) # values and argsort   (*Xs, Nx, n)
        sX_inv = torch.argsort(sX, dim= -2)   # inverse permutations (*Xs, Nx, n)
        Yp_sort = torch.sort(Yp, dim= -2).values # (*Ys, Ny, n)

        sX_inv  = unsqueeze(sX_inv , nXs, nYs) # (   *Xs   , 1, ..., 1, Nx, n)
        Xp_sort = unsqueeze(Xp_sort, nXs, nYs) # (   *Xs   , 1, ..., 1, Nx, n)
        Yp_sort = unsqueeze(Yp_sort, 0  , nXs) # (1, ..., 1,   *Ys    , Ny, n)

        sX_inv = sX_inv.unsqueeze(-2)                                # (   *Xs   , 1, ..., 1, Nx, 1, n)
        Xp_sort = Xp_sort.unsqueeze(-2)                              # (   *Xs   , 1, ..., 1, Nx, 1, n)
        Yp_sort = Yp_sort[(slice(None),)*(nXs + nYs) + (sweep_idx,)] # (1, ..., 1,    *Ys   , Nx, k, n)
        sweep_red = sweep_red.unsqueeze(-1)                                                # (Nx, k, 1)

        # Unsort sweep_red. We could also return return sX_inv as output and only unsort the gradient.
        sweep_red = unsqueeze(sweep_red, 0, nXs + nYs)              # (1, ..., 1, 1, ..., 1, Nx, k, 1)
        sweep_red = torch.take_along_dim(sweep_red, sX_inv, dim=-3) # (   *Xs   , 1, ..., 1, Nx, k, n)
        diff = Xp_sort - Yp_sort                                    # (   *Xs   ,    *Ys   , Nx, k, n)
        diff = torch.take_along_dim(diff, sX_inv, dim=-3)           # (   *Xs   ,    *Ys   , Nx, k, n)
        return diff,  sweep_red

    # def align_sliced_diff_avg(self, psis, X, Y):
    #     """Computes the differences between the aligned projections of points in X and points in Y.
    #     Aligned means we ordonate the points before computing the differences. X and Y can contain several
    #     sets of points, in which case the operation is done for every pair of sets. #! Rewrite

    #     Args:
    #         psis (Tensor): Shape (n, *ms). References on the manifold, with which we define the projections (slicing operation)
    #         X (Tensor): Shape (*Xs, Nx, *ms). Points (or set(s) of points) on the manifold.
    #         Y (Tensor): Shape (*Ys, Ny, *ms). Points (of set(s) of points) on the manifold.

    #     Returns:
    #         diff (Tensor): Shape (*Xs, *Ys, Nx, n). Aligned differences of the projections.
    #     """
    #     Xp = self.sman.operator(psis, X) # (*Xs, Nx, n) # X projected
    #     Yp = self.sman.operator(psis, Y) # (*Ys, Ny, n)
    #     nXs = len(Xp.shape) - 2
    #     nYs = len(Yp.shape) - 2
    #     Nx = Xp.shape[-2]
    #     Ny = Yp.shape[-2]

    #     # Comments below suppose Xs=Ys=(). When it is not the case, the operation is done for every entry.
    #     sX     = torch.argsort(Xp, dim= -2) # each column q contains \sigma_{X,\psi_q}      #(*Xs, Nx, n)
    #     sX_inv = torch.argsort(sX, dim= -2) # each column q contains \sigma^{-1}_{X,\psi_q} #(*Xs, Nx, n)
    #     Yp_sort = torch.sort(Yp, dim= -2).values #(*Ys, Ny, n)

    #     sX_inv  = unsqueeze(sX_inv , nXs, nYs) # (  *Xs    , 1, ..., 1, Nx, n)
    #     Xp      = unsqueeze(Xp     , nXs, nYs) # (  *Xs    , 1, ..., 1, Nx, n)
    #     Yp_sort = unsqueeze(Yp_sort, 0  , nXs) # (1, ..., 1,   *Ys    , Ny, n)

    #     sweep = sweep_matrix(Nx, Ny) # (Nx, Ny)
    #     Yp_aligned = torch.tensordot(Yp_sort, sweep, dims=([-2], [1])) # (1, ..., 1, *Ys, n, Nx) # aligned with Xp sorted
    #     Yp_aligned = torch.transpose(Yp_aligned, -2, -1)               # (1, ..., 1, *Ys, Nx, n)
    #     Yp_aligned = torch.take_along_dim(Yp_aligned, sX_inv, dim=-2)  # (   *Xs   , *Ys, Nx, n) # aligned with Xp
    #     diff = Xp - Yp_aligned                                         # (   *Xs   , *Ys, Nx, n)
    #     return diff
    
    def sliced_dist_square_gnrl(self, psis, X, Y):
        """Computes the squared sliced Wasserstein distance between the emprical measures represented by the points
        in X and Y, for the different slices psis. X and Y can contain different sets of points, each one representing
        a measure, in which case, the distances are computed for every pair of sets.

        Args:
            psis (Tensor): Shape (n, *ms). Points on the manifold representing the different slices.
            X (Tensor): Shape (*Xs, Nx, *ms). Points (or sets of points) representing empirical measure(s).
            Y (Tensor): Shape (*Ys, Ny, *ms). Points (or sets of points) representing empirical measure(s).

        Returns:
            dist (Tensor): Shape (*Xs, *Ys). Squared sliced distance between the two measures (or every pairs of measures).
        """
        n = psis.shape[0]
        diff, sweep_red = self.align_sliced_diff_gnrl(psis, X, Y) #(*Xs, *Ys, Nx, k, n), (Nx, k, 1)
        square = diff**2 * sweep_red
        return torch.sum(square, dim=(-3, -2, -1))/n
    
    def sliced_dist_square(self, psis, X, Y):
        """Computes the squared sliced Wasserstein distance between the emprical measures represented by the points
        in X and Y, for the different slices psis. X and Y can contain different sets of points, each one representing
        a measure, in which case, the distances are computed for every pair of sets.

        Args:
            psis (Tensor): Shape (n, *ms). Points on the manifold representing the different slices.
            X (Tensor): Shape (*Xs, Nx, *ms). Points (or sets of points) representing empirical measure(s).
            Y (Tensor or list): Tensor of shape (*Ys, Ny, *ms) or list of length M of tensors of shapes (Ny_j, *ms). 
                Points (or sets of points) representing empirical measure(s).

        Returns:
            dist (Tensor): Shape (*Xs, *Ys) (with Ys=(M,) when Y is a list). Squared sliced distance between the two
                measures (or every pairs of measures).
        """
        if type(Y) == list:
            return torch.stack([self.sliced_dist_square_gnrl(psis, X, Yj) for Yj in Y], dim=-1)
        else:
            Nx = X.shape[-self.sman.ndim-1]
            Ny = Y.shape[-self.sman.ndim-1]
            if Nx != Ny:
                return self.sliced_dist_square_gnrl(psis, X, Y)
            else:
                return self.sliced_dist_square_same(psis, X, Y)
        
    def sliced_loss(self, psis, X, Y, lambdas=None):
        """Returns the value of the Frechet-like loss associated to the sliced wasserstein distance.

        Args:
            psis (Tensor): Shape (n, *ms). Points on the manifold representing the different slices.
            X (Tensor): Shape (*Xs, Nx, *ms). Points (or sets of points) representing empirical measure(s).
            Y (Tensor or list): Tensor of shape (M, Ny, *ms) or list of length M of tensors of shapes (Ny_j, *ms). 
                Points (or sets of points) representing empirical measure(s).
            lambdas (tensor | None). Shape (M,). Coefficients for the barycentric~ objective.

        Returns:
            loss (float): Frechet-like loss associated to the sliced wasserstein distance
        """
        square_distances = self.sliced_dist_square(psis, X, Y)
        M = len(Y)
        lambdas = 1/M * torch.ones((M,)) if lambdas is None else lambdas
        return torch.tensordot(square_distances, lambdas, dim=1)

    def functional_grad_gnrl(self, psis, X, Y, lambdas, debug=False, it=None, **kwargs):
        """Compute the gradient of the Frechet functional (the one we have to minimise to get the Frechet barycentre) using the 
        sliced Wasserstein distance.

        Args:
            psis (Tensor): Shape (n, *ms). Points on the manifold representing the different slices.
            X (Tensor): Shape (Nx, *ms). Points representing the empirical measure with which we sequentially approximate the barycentre.
            Y (list of Tensors): Length M. Tensors have shapes (Ny_j, *ms). Sets of points representing the empirical measures of 
                which we want to compute the barycentre.
            lambdas (Tensor): Shape (M,). Weights of the different measure in the barycentre.

        Returns:
            gradient (Tensor): Shape (N, *ms). gradient of the Frechet functional in X.
            loss (float | None): The value of the Frechet functional.
        """
        n = psis.shape[0]
        N = X.shape[0]
        diffs_plans = [self.align_sliced_diff_gnrl(psis, X, Yj) for Yj in Y] # of shapes (Nx, k_j, n), (Nx, k_j, 1)
        
        square_distances = [torch.sum(diff**2 * sweep_red)/n for diff, sweep_red in diffs_plans] # length M
        loss = lambdas.dot(torch.tensor(square_distances))

        # gradients in the slices
        gdiff = [2 * torch.sum(diff * sweep_red, dim=1)/n for diff, sweep_red in diffs_plans] # length M, shapes (Nx, n)
        gdiff = torch.stack(gdiff) # (M, Nx, n)
        # Note the difference with the same method in SWBarycenter. gdiff already contains the factor 2/(Nx*n) (1/Nx being 
        # contained in sweep_red).

        gf = self.sman.operator_gradient_factor(psis, X, **kwargs) # (Nx, n)
        gradient_per_Yi = torch.tensordot(gdiff * gf, psis, dims=([2], [0])) # (M, Nx, *ms)
        gradient = torch.tensordot(lambdas, gradient_per_Yi, dims=1) # (Nx, *ms)

        return gradient, loss

    def stop_criterion(self):
        """Stopping criterion based on the slope of the loss, its uncertainty, and the slope of the step_norm.

        Returns:
            bool: Whether to stop the algorithm.
        """
        l_min = 40
        eps = 0.05
        l = len(self.L_loss)
        if l < l_min:
            return False
        reg_loss0 = linregress(np.arange(l//3), self.L_loss[:l//3], alternative='less')
        reg_loss1 = linregress(np.arange(l//2), self.L_loss[-(l//2):], alternative='less')
        stopped_descending = reg_loss0.slope + 2*reg_loss0.stderr < 0
        stopped_descending &= reg_loss1.slope - 2*reg_loss1.stderr > eps*(reg_loss0.slope + reg_loss0.stderr)

        reg_step0 = linregress(np.arange(l//3), self.L_step[:l//3])
        reg_step1 = linregress(np.arange(l//3), self.L_step[-(l//3):])
        stopped_descending &= reg_step1.slope > 0.01*min(0, reg_step0.slope)
        self.L_stop.append([reg_step1.slope, reg_step1.slope - reg_step1.stderr])
        return stopped_descending


    # Stochastic gradient descent loop
    def fit(self, Y, lambdas, X0, n_psis, tau_init, max_iter, fixed_psis=False, a_tau=1, pow_tau=0, tqdm_leave=True, stop=True, **kwargs):
        """Computes the Frechet barycentre of emprical measures, using the sliced Wasserstein distance.

        Args:
            Y (Tensor | list of tensors): Tensor of shape (M, N, *ms) or list of length M, of tensors of shapes (N_j, *ms).
                Sets of points representing the M measures of which we want to compute the barycentre.
                Complexity is bigger in the case of a list.
            lambdas (Tensor): Shape (M,). Weights of the barycentre.
            X0 (Tensor): Shape (N, *ms). Source measure (initial value of the approximating sequence).
            n_psis (int): Number of slices
            tau_init (float): Learning rate or initial learning rate.
            max_iter (int): Number of iterations.
            fixed_psis (bool | int): False: psis are randomly samples at each step. True: psis are randomly sampled once, 
                then fixed. n: n psis are fixed, and at each iteration, n_psis among them are chosen.
            a_tau (float): scale parameter of the tau decrease. Defaults to 1.
            pow_tau (float): exponent in the tau decrease. Defaults to 0.
            tqdm_leave (bool): whether to leave the tqdm bar after the end. Useful when several experiments are done in a row.\
                Defaults to True.
            stop (bool): whether to stop the algorithm earlier, based on a convergence criterion.

        The law of the tau decrease is given by tau_k = tau_init / (1 + k*a_tau)**pow_tau

        Computes:
            self.barycenter (Tensor): Points supporting the free-support PSB.
            self.frechet_quantity (float): final loss value
            self.L_loss (list): list of loss values
            self.L_step (list): list of step norms \|X_k - X_{k-1}\|

        Returns: self
        """
        X = torch.clone(X0)
        M = len(Y)
        gnrl = type(Y) == list
        lambdas = lambdas if lambdas is not None else 1/M * torch.ones((M,))
        self.measures = Y
        self.lambdas = lambdas
        self.step_save = max(max_iter//self.number_save, 1)

        self.barycenter = X
        self.frechet_quantity = torch.inf

        # Preparing slices if needed
        if fixed_psis is True:
            psis = self.sman.sample_uniform(n_psis) # (n_psis, *ms)
        if type(fixed_psis) == int:
            psis_0 = self.sman.sample_uniform(fixed_psis)
        
        for it, tau in zip(tqdm(range(max_iter), leave=tqdm_leave), tau_generator(tau_init, a_tau, pow_tau)):

            # Choosing slices
            if fixed_psis is False:
                psis = self.sman.sample_uniform(n_psis)
            elif type(fixed_psis) == int:
                psis = psis_0[torch.randint(fixed_psis, size=(n_psis,))]

            # Computing gradient and loss
            if gnrl:
                gradient, loss = self.functional_grad_gnrl(psis, X, Y, lambdas, it=it, **kwargs)
            else:
                gradient, loss = self.functional_grad(psis, X, Y, lambdas, it=it, **kwargs)

            # Saving best found measure
            if loss < self.frechet_quantity:
                self.barycenter = X
                self.frechet_quantity = loss

            # Gradient step
            # X = self.projection_manifold(X - tau*gradient)
            # X = self.projection_manifold(X - tau*self.projection_tangent_space(X, gradient))
            X_new = self.sman.exponential(X, - tau*self.sman.projection_tangent_space(X, gradient))
            X_new = self.sman.projection_manifold(X_new) # seems necessary

            # Saving results
            self.L_loss.append(loss)
            step_norm = torch.norm(X_new - X)
            self.L_step.append(step_norm)
            if it%self.step_save == 0:
                self.L.append(X)

            X = X_new

            if stop and self.stop_criterion():
                break

        return self
    
    def plot_samples(self, depthshade=True):
        self.sman.plot_samples(self.measures, self.barycenter, depthshade=depthshade)



class ESWBarycenter():
    """Class implementing the fixed-support Sliced Wasserstein Barycenter algorithm. Each instance is associated \
        to a specific manifold and slicing operator."""
    def __init__(self, sman:SlicedManifold):
        """
        Args:
            sman (SlicedManifold): sliced manifold
        """
        self.sman = sman
        self.support = None
        self.measures = None
        self.lambdas = None
        self.barycenter = None
        self.frechet_quantity = None

        self.L_loss = []
        self.L_step = []
        self.L = []
        self.number_save = 20
        self.step_save = None

        self.L_stop = []
    
    @property
    def it_save_range(self):
        return torch.arange(len(self.L))*self.step_save
    

    def functional_grad(self, psis, X, W, V, lambdas, it=None):
        """Computes the gradient of the (Frechet-like) loss function.

        Args:
            psis (Tensor): Shape (n, *ms). Slices (Direction of projections).
            X (Tensor): Shape (N, *ms). Coordinates of the support points.
            W (Tensor): Shape (N). Weights of the candidate measure.
            V (Tensor): Shape (M, N). Weights of the input measures.
            lambdas (Tensor): Shape (M,). Weights in the loss definition.
            it (int | None, optional): iteration. For debug purpose. Defaults to None.

        Returns:
            grad_final (Tensor): Shape (N,). Gradient.
            loss (float): loss value.
        """
        N = len(X)
        M = len(V)
        n = len(psis)

        # Projection (slicing) of the support points
        Xp = X @ psis.T # N, n_psis

        # Sorting the support points
        Xp_sort, sX = torch.sort(Xp, dim=0)
        W_sort = torch.take_along_dim(W[:, None], sX, dim=0) # (N, n)
        V_sort = torch.take_along_dim(V[:, :, None], sX[None], dim=1) # (M, N, n)

        # Computation of the cumulative function and merging
        Wc = torch.cumsum(W_sort, dim=0)[:-1] # we remove the last value, which is 1 (useless)
        Vc = torch.cumsum(V_sort, dim=1)[:, :-1]
        WV_merge = torch.cat([Wc[None, :,:].repeat(M, 1, 1), Vc], dim=1) # (M, 2N-2, n)
        mask_w = torch.cat([torch.ones ((M, N-1, n), dtype=torch.long), torch.zeros((M, N-1, n), dtype=torch.long)], dim=1)
        mask_v = torch.cat([torch.zeros((M, N-1, n), dtype=torch.long), torch.ones ((M, N-1, n), dtype=torch.long)], dim=1)
        
        # Sorting the merged cumulative functions
        WV_sort, sVW = torch.sort(WV_merge, dim=1)           # (M, 2N-2, n)
        mask_w = torch.take_along_dim(mask_w, sVW, dim=1)    # (M, 2N-2, n)
        mask_v = torch.take_along_dim(mask_v, sVW, dim=1)    # (M, 2N-2, n)
        
        # Computing the difference in cumulative function and the difference in support points,
        # i.e. the step in the abscissa of the icfs and the difference in icfs (where icf = inverse of the cumulative function)
        h = torch.diff(WV_sort, dim=1)                   # (M, 2N-3, n)
        idx1 = torch.cumsum(mask_w, dim=1)[:, :-1, :]    # (M, 2N-3, n)
        idx2 = torch.cumsum(mask_v, dim=1)[:, :-1, :]    # (M, 2N-3, n)
        Xp_us = torch.unsqueeze(Xp_sort, 0)              # (1, N, n)
        q = torch.take_along_dim(Xp_us, idx1, dim=1) - torch.take_along_dim(Xp_us, idx2, dim=1) # (M, 2N-3, n)

        # Objective (loss)
        loss_j = torch.sum(q**2 * h, dim=(1, 2))/n # (M,)
        loss = torch.dot(lambdas, loss_j)   # scalar

        # Computation of the gradient with respect to the cumulative function vector
        q_long = torch.cat([torch.zeros((M, 1, n)), q, torch.zeros((M, 1, n))], dim=1) # (M, 2N-1, n)
        sVW_inv = torch.argsort(sVW, dim=1)[:, :N-1, :] # (M, N-1, n)
        grad_plus  = torch.take_along_dim(q_long, sVW_inv    , dim=1) ** 2   # (M, N-1, n)
        grad_minus = torch.take_along_dim(q_long, sVW_inv + 1, dim=1) ** 2   # (M, N-1, n)
        grad_cum = torch.tensordot(lambdas, grad_plus - grad_minus, dims=1)     # (N-1, n)

        # Computation of the gradient with respect to the weights
        # pseudo_grad = torch.diff(grad0, prepend=torch.zeros((1,)), append=torch.ones((1,)), dim=1)
        grad = torch.flip(torch.cumsum(torch.flip(grad_cum, [0]), 0), [0]) # (N-1, n)
        grad = torch.cat([grad, torch.zeros((1, n))], dim=0)                 # (N, n)

        sX_inv = torch.argsort(sX, dim=0)
        grad_unsort = torch.take_along_dim(grad, sX_inv, dim=0)

        grad_final = torch.sum(grad_unsort, dim=1)/n

        return grad_final, loss

    def proj_tang(self, vect):
        """Projection on the orthogonal space of vector (1, ..., 1), i.e. the tangiental space of the \
        probability simplex.

        Args:
            vect (Tensor): Shape (N). Vector to project.

        Returns:
            Tensor: Projected vector.
        """
        # (N,)
        N = len(vect)
        u = torch.ones(N)/sqrt(len(vect))
        return vect - vect.dot(u)*u

    def proj_simplex(self, vect):
        """Projection on the probability simplex, using algorithm from [1].

        [1] Wand, W., Carreira-Perpiñan, M. A., Projection onto the probability \
        simplex: An efficient algorithm with a simple proof, and an application

        Args:
            vect (Tensor): Vector to be projected.

        Returns:
            Tensor: Projection.
        """
        # 
        N = len(vect)
        v = torch.sort(vect, descending=True).values
        vcs = torch.cumsum(v, 0)
        a = torch.arange(1, N+1)
        new_var = (v + 1/a * (1 - vcs))
        mask = new_var > 0
        rho = a[mask][-1] #maximum index
        lambd = 1/rho*(1 - vcs[rho - 1])
        return torch.maximum(vect + lambd, torch.zeros(()))


    def stop_criterion(self):
        """Stopping criterion based on the slope of the loss, its uncertainty, and the slope of the step_norm.

        Returns:
            bool: Whether to stop the algorithm.
        """
        l_min = 40
        eps = 0.05
        l = len(self.L_loss)
        if l < l_min:
            return False
        reg_loss0 = linregress(np.arange(l//3), self.L_loss[:l//3], alternative='less')
        reg_loss1 = linregress(np.arange(l//2), self.L_loss[-(l//2):], alternative='less')
        stopped_descending = reg_loss0.slope + 2*reg_loss0.stderr < 0
        stopped_descending &= reg_loss1.slope - 2*reg_loss1.stderr > eps*(reg_loss0.slope + reg_loss0.stderr)

        reg_step0 = linregress(np.arange(l//3), self.L_step[:l//3])
        reg_step1 = linregress(np.arange(l//3), self.L_step[-(l//3):])
        stopped_descending &= reg_step1.slope > 0.01*min(0, reg_step0.slope)
        self.L_stop.append([reg_step1.slope, reg_step1.slope - reg_step1.stderr])
        return stopped_descending

    # Stochastic gradient descent loop
    def fit(self, X, V, lambdas, W0, n_psis, tau_init, max_iter, fixed_psis=False, a_tau=1, pow_tau=0, tqdm_leave=True, stop=True, **kwargs):
        """computes the fixed-support Sliced Wasserstein Barycenter using a stochastic gradient descent algorithm.

        Args:
            X (Tensor): Shape (N, *ms). Coordinates of the points of the fixed support.
            V (Tensor): Shape (M, N). Weights of the measures for which to compute the barycentre.
            lambdas (Tensor): Shape (M,). Coefficients in the barycentre.
            W0 (Tensor): Shape (N,). Initial point (expressed with weights).
            n_psis (int): Number of slices (i.e. of projections).
            tau_init (float): Initial value of the learning rate.
            max_iter (int): Number maximum of iterations.
            fixed_psis (bool|int, optional): Whether we fix the direction of the slices in advance. If True slices are just fixed.
                If int, the integer gives the number of fixed directions and n_psis directions are taken from them at each iteration. 
                Defaults to False.
            a_tau (float, optional): coefficient in the tau decrease. Defaults to 1.
            pow_tau (float, optional): Power in the tau decrease. Defaults to 0.
            tqdm_leave (bool, optional): Whether to leave the tqdm bar at the end of computations. Defaults to True.
            stop (bool, optional): Whether to use the stop criterion to stop the algorithm in advance. Defaults to True.

        Returns:
            self
        """
        W = torch.clone(W0)
        M = len(V)
        lambdas = lambdas if lambdas is not None else 1/M * torch.ones((M,))
        self.support = X
        self.measures = V
        self.lambdas = lambdas
        self.step_save = max(max_iter//self.number_save, 1)

        self.barycenter = W
        self.frechet_quantity = torch.inf

        # Preparing slices if needed
        if fixed_psis is True:
            psis = self.sman.sample_uniform(n_psis) # (n_psis, *ms)
        if type(fixed_psis) == int:
            psis_0 = self.sman.sample_uniform(fixed_psis)
        
        for it, tau in zip(tqdm(range(max_iter), leave=tqdm_leave), tau_generator(tau_init, a_tau, pow_tau)):

            # Choosing slices
            if fixed_psis is False:
                psis = self.sman.sample_uniform(n_psis)
            elif type(fixed_psis) == int:
                psis = psis_0[torch.randint(fixed_psis, size=(n_psis,))]

            # Computing gradient and loss
            gradient, loss = self.functional_grad(psis, X, W, V, lambdas, it=it, **kwargs)

            # Saving best found measure
            if loss < self.frechet_quantity:
                self.barycenter = W
                self.frechet_quantity = loss

            # Gradient step
            W_new = self.proj_simplex(W - tau*self.proj_tang(gradient))

            # Saving results
            self.L_loss.append(loss)
            step_norm = torch.norm(W_new - W)
            self.L_step.append(step_norm)
            if it%self.step_save == 0:
                self.L.append(W)

            W = W_new

            if stop and self.stop_criterion():
                break

        return self
    


# Wasserstein barycentre. Only on the sphere !!! #! Change this
class WBarycenter():
    """Computes the Wasserstein barycenter on the sphere using POT library."""
    def __init__(self):
        self.measures = None
        self.lambdas = None
        self.barycenter = None
        self.support = None

    def fit(self, X, V, lambdas, verbose=False):
        self.measures = V
        self.support = X

        M = len(V)
        lambdas = lambdas if lambdas is not None else 1/M * torch.ones((M,))
        self.lambdas = lambdas

        euclidean_distances = torch.norm(X[:, None] - X[None, :], dim=2)
        spherical_distances = 2 * torch.arcsin(euclidean_distances / 2)
        costs = spherical_distances ** 2
        self.barycenter = ot.lp.barycenter(V.T, costs, lambdas, verbose=verbose)
        return self

class WRegBarycenter():
    """Computes the regularised Wasserstein barycenter on the sphere using POT library."""
    def __init__(self):
        self.measures = None
        self.lambdas = None
        self.barycenter = None
        self.support = None
    
    def fit(self, X, V, lambdas, reg, verbose=False):
        self.measures = V
        self.support = X

        M = len(V)
        lambdas = lambdas if lambdas is not None else 1/M * torch.ones((M,))
        self.lambdas = lambdas

        euclidean_distances = torch.norm(X[:, None] - X[None, :], dim=2)
        spherical_distances = 2 * torch.arcsin(euclidean_distances / 2)
        costs = spherical_distances ** 2
        self.barycenter = ot.bregman.barycenter(V.T, costs, reg, lambdas, verbose=verbose)
        return self

