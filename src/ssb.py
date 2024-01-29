import torch
from torch.nn import functional as F

from src.barycenters import tau_generator, LSWBarycenter

import sys
sys.path.append("Spherical_Sliced-Wasserstein/lib")
from sw_sphere import sliced_wasserstein_sphere

from utils import tqdm, USE_TQDM

device = torch.device("cpu")

class SSWBarycenter(LSWBarycenter):
    """Semi-circular Sliced Wasserstein Barycenter. Adapts and calls the code of [1].
    
    [1] C. Bonet, P. Berg, N. Courty, F. Septier, L. Drumetz, and M. T. Pham, “Spherical Sliced-Wasserstein,” \
        presented at the The Eleventh International Conference on Learning Representations, Sep. 2022. \
        Available: https://openreview.net/forum?id=jXQ0ipgMdU
    """
    def __init__(self):
        super().__init__(sman=None)

    def fit(self, Y, lambdas, X0=None, n_psis=500, tau_init=100, max_iter=1000, exp=True, a_tau=1, pow_tau=0, tqdm_leave=True, stop=True):
        """Computes the barycenter of measures in Y.

        Args:
            Y (Tensor | list): contains M tensors of sizes (Nj, 3) which are samples of 
                the measures we want to compute the barycentre of

            lambdas (Tensor): size (M,). Coefficients for the different measures in the barycentre.
            X0 (Tensor, optional): size (N, 2). Initial sample. Defaults to None.
            max_iter (int, optional): Number of iteration. Defaults to 1000.
            n_psis (int, optional): Number of slices (projections). Defaults to 500.
            tau (int, optional): Learning rate in the gradient descent step. Defaults to 100.
            exp (bool, optional): Whether we consider the exponential of the gradient (True) 
                or its projection (False). Defaults to True.
        """
        M = len(Y)
        if lambdas is None:
            lambdas = 1/M * torch.ones((M,))
        self.lambdas = lambdas
        self.measures = Y

        if X0 is None:
            X0 = torch.randn((100, 3), device=device)
            X0 = F.normalize(X0, p=2, dim=-1)
        else:
            X0 = torch.clone(X0)
            
        X0.requires_grad_(True)
        

        self.L.append(X0.clone())

        pbar = tqdm(range(max_iter), leave=tqdm_leave)
        if USE_TQDM[0]: pbar.set_postfix_str(f"loss = ?")

        for k, tau in zip(pbar, tau_generator(tau_init, a_tau, pow_tau)):
            grads = []
            for Yj in Y:            
                sw = sliced_wasserstein_sphere(Yj, X0, n_psis, device, p=2)
                gradj = torch.autograd.grad(sw, X0)[0]
                grads.append(gradj)
            grads = torch.stack(grads)
            grad = torch.tensordot(lambdas, grads, dims=1)

            if exp:
                grad -= torch.sum(X0*grad, dim=-1)[:, None]*X0
                v = - tau * grad
                norm_v = torch.linalg.norm(v, axis=-1)[:,None]
                X_new = X0 * torch.cos(norm_v) + torch.sin(norm_v) * v/norm_v
            else:
                X_new = X0 - tau * grad
                X_new = F.normalize(X_new, p=2, dim=1)
            step_norm = torch.norm(X_new - X0).detach()
            X0 = X_new
            
            self.L_step.append(step_norm)
            self.L_loss.append(sw.item())
            self.L.append(X0.clone().detach())
            if USE_TQDM[0]: pbar.set_postfix_str(f"loss = {sw.item():.3f}")

            if stop and self.stop_criterion():
                break
        
        self.barycenter = X0
        self.frachet_quantity = sw.item()