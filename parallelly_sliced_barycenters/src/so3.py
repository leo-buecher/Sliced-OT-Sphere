import torch
import numpy as np
from scipy.stats import vonmises_fisher
from scipy.spatial.transform import Rotation

from src.manifold import Manifold, SlicedManifold, ScaManifold

from itertools import product as iter_prod
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

class SO3(Manifold):
    """Class implementing the main functions corresponding to the SO3 manifold.
    It is designed for cooperative inheritance, along with a class using those operations."""
    def __init__(self):
        super().__init__()
        self.ms = (3, 3)
    
    def projection_tangent_space(self, X, dX):
        """Projects a vector onto the tangential space
        
        Args:
            X (Tensor): Shape (*Ns, *ms). The vectors in which to consider the tangential spaces.
            dX (Tensor): Shape (*Ns, *ms). The vectors to project to the tangential spaces. (We actually
                project X + dX)
        
        Returns:
            dX_p (Tensor): Shape (*Ns, *ms). The vectors belonging to the tangential spaces.
        """
        PdX = 1/2 * (dX - X @ torch.transpose(dX, -2, -1) @ X)
        return PdX

    def cayley_retraction(self, X, dX):
        """Cayley retraction of dX (tangent vector in X for SO3) in X on SO3

        Args:
            X (Tensor): shape *Ns, 3, 3. Rotation matrices
            dX (Tensor): shape *Ns, 3, 3. Tangent vectors of SO3 in the points given by X.

        Returns:
            (Tensor): shape *Ns, 3, 3. Each is the cay retraction on SO3
        """
        # W = X.transpose(0, 2, 1) @ dX
        CayW = torch.linalg.inv(X - 1/2 * dX) @ (X + 1/2 * dX)
        return X @ CayW

    def exponential(self, X, dX):
        """Takes a vector from the tangential space to the manifold

        Args:
            X (Tensor): Shape (*Ns, *ms). The vectors of reference.
            dX (Tensor): Shape (*Ns, *ms). The tangential vectors, as produced by self.projection_tangent_space
        
        Returns:
            X_new (Tensor): Shape (*Ns, *ms). The new vectors of the manifold corresponding to the 
                exponential of each dX in each X.
        """
        return X @ torch.linalg.matrix_exp(torch.transpose(X, -2, -1) @ dX)
        
    
    def projection_manifold(self, X):
        """Projects a vector on the manifold
        
        Args:
            X (Tensor): Shape (*Ns, *ms). The vectors to project on the manifold.
        
        Returns:
            X_new (Tensor): Shape (*Ns, *ms). The projected vectors.
        """
        U, S, Vh = torch.linalg.svd(X) #SVD of each vector in X
        return U @ Vh #matmul of each (U[i_1, ..., i_k], Vh[i_1, ..., i_k]) where k=len(Xs)
    
    def axis_angle_to_matrix(self, axes, angles):
        """Generates rotation matrices corresponding to the given axes and teh given angles

        Args:
            axes (Tensor): rotation axes. Shape (*Ns, 3)
            angles (Tensor): angles. Shape Ns

        Returns:
            (Tensor): Shape (*Ns, 3, 3), rotation matrices
        """
        c = torch.cos(angles)
        s = torch.sin(angles)
        Ns = angles.shape
        batch_ndim = len(Ns)

        axes = torch.permute(axes, (-1, *range(batch_ndim))) # (3, *Ns)

        t1 = axes[:,None]*axes[None,:] * (1-c) # (3, 3, *Ns)
        t2 = torch.stack([torch.stack([    c     , -axes[2]*s,  axes[1]*s]),
                          torch.stack([ axes[2]*s,     c     , -axes[0]*s]),
                          torch.stack([-axes[1]*s,  axes[0]*s,     c     ])]) # (3, 3, *Ns)
        t = t1 + t2
        t = torch.permute(t, (*range(-batch_ndim, 0), 0, 1))
        return t

    def cano_angle_to_matrix(self, axis, angles):
        """Generates rotation matrices corresponding to the given angles, for a given fixed canonical axis

        Args:
            axis (int): 0, 1, 2. The rotation axis.
            angles (torch.ndarray | float): angles. Shape Ns

        Returns:
            torch.ndarray(*Ns, 3, 3): rotation matrices
        """
        c = torch.cos(angles)
        s = torch.sin(angles)
        Ns = angles.shape
        batch_ndim = len(Ns)

        t = torch.zeros((3, 3, *Ns))
        t[0, 0] = 1
        t[[1, 2], [1, 2]] = c
        t[2, 1] = s
        t[1, 2] = -s

        # one = torch.ones(Ns)
        # zer = torch.zeros(Ns)
        # t = torch.tensor([[one, zer, zer], [zer, c, -s], [zer, s, c]]) # (3, 3, *Ns)

        perm = (torch.arange(3) - axis) % 3
        t = t[perm[:, None], perm[None, :]]

        t = torch.permute(t, (*range(-batch_ndim, 0), 0, 1)) # (*Ns, 3, 3)
        return t

    def parametrisation(self, A):
        """Returns a vector on the manifold from a vector in R^d
        
        Args:
            A (Tensor): Shape (d, *As) where d is the dimension of the coordinate space. Coordinates of
                parametrised points
        
        Returns:
            X (Tensor): Shape (*As, *ms). Points on the manifold.
        """
        alphas, betas, gammas = A
        r3a = self.cano_angle_to_matrix(2, alphas)
        r2b = self.cano_angle_to_matrix(1, betas)
        r3g = self.cano_angle_to_matrix(2, gammas)
        return r3a @ r2b @ r3g
    
    def sample_uniform(self, n):
        """Returns a sample from the uniform probability distribution on the manifold
        
        Args:
            n (int): number of samples
        
        Returns:
            X (Tensor): Shape (n, *ms). Points on the manifold, uniformely distributed.
        """
        # xhis = np.random.randn(n, 3, 3)
        # Qs, _ = np.linalg.qr(xhis)
        # # It seems that the determinant is always +1
        # return Qs

        alphas = torch.rand(n)* 2 * torch.pi
        betas  = torch.arccos(torch.rand(n)* 2 - 1)
        gammas = torch.rand(n)* 2 * torch.pi
        A = torch.stack([alphas, betas, gammas])
        return self.parametrisation(A)
    
    def sample_uniform_portion(self, bounds, n):
        """Returns a sample from the uniform probability distribution on a portion corresponding to a 
        parallelepipedeon in the coordinate space.
        
        Args:
            bounds (Tensor): Shape (d, 2) where d is the dimension of the coordinate space. 
                The bounds (min, max) for each of the d coordinates.
            n (int): number of samples
        
        Returns:
            X (Tensor): Shape (n, *ms). Points on the manifold uniformely distributed in the portion 
                described by bounds.
        """
        a_min, a_max = bounds[0]
        cb_max, cb_min = torch.cos(bounds[1])
        g_min, g_max = bounds[2]

        alphas = torch.rand(n)*(a_max - a_min) + a_min
        betas  = torch.arccos(torch.rand(n)*(cb_max - cb_min) + cb_min)
        gammas = torch.rand(n)*(g_max - g_min) + g_min

        A = torch.stack([alphas, betas, gammas])
        return self.parametrisation(A)
    
    def sample_vMF_quat(self, coords, kappas, n):
        """Samples one or several distributions on SO(3) corresponding to von-Mises-Fisher distributions\
        on the unit sphere of quaternions.

        Args:
            coords (Sequence): Shape (*ms, 3, 3). Mean directions of the distributions, expressed as rotation matrices.
            kappas (Sequence): Shape (*ms,). Concentration parameters of the distributions.
            n (int): Number of samples in each distribution.

        Returns:
            Tensor: Shape (*ms, 3, 3). Samples, expressed as rotations matrices.
        """
        (*ms, _, _) = coords.shape
        if not type(kappas)== torch.Tensor:
            kappas = torch.tensor(kappas)
        kappas = torch.broadcast_to(kappas, ms)
        
        X = torch.zeros((*ms, n, 3, 3))
        for js in iter_prod(*[range(m) for m in ms]):
            mu_quat = Rotation.from_matrix(coords[*js]).as_quat()
            samples_quat = torch.tensor(vonmises_fisher(mu_quat, float(kappas[*js])).rvs(n))
            X[*js] = torch.tensor(Rotation.from_quat(samples_quat).as_matrix())
        return X

    def identify_eulers_angles(self, rotations):
        """Identifies the euler angles of given rotations.

        Args:
            rotations (Tensor): Shape (*Ns, 3, 3)
        
        Returns:
            (Tensor): Shape (3, *Ns). Euler angles.
        """
        
        betas = torch.arccos(rotations[..., 2, 2])
        sin_betas = torch.sin(betas)

        cos_a = rotations[..., 0, 2]/sin_betas
        sin_a = rotations[..., 1, 2]/sin_betas
        sign_sin_a = (sin_a>=0).long() * 2 - 1
        alphas = sign_sin_a*torch.arccos(cos_a) + (1 - sign_sin_a)*torch.pi
        
        cos_g = - rotations[..., 2, 0]/sin_betas
        sin_g =   rotations[..., 2, 1]/sin_betas
        sign_sin_g = (sin_g>=0).long() * 2 - 1
        gammas = sign_sin_g*torch.arccos(cos_g) + (1 - sign_sin_g)*torch.pi
        
        A = torch.stack([alphas, betas, gammas])
        return A
    
    def plot_samples_2(self, Y, X=None, depthshade=True):
        """Represents samples on the manifold
        
        Args:
            Y (Tensor): Shape (M, N, *ms). Target measures.
            X (Tensor): Shape (N, *ms). Other empirical measure.
            depthshade (bool): Whether or not to change the color of the points according to their
                depth in the 3D plot space.
        """

        cmap = LinearSegmentedColormap.from_list("mycmap", ["blue", "magenta", "red", "yellow", "green", "cyan", "blue"])
        cm_norm = Normalize(0, 2*np.pi, clip=True)
        mappable = ScalarMappable(norm=cm_norm, cmap=cmap) # Associate a map (value -> [0, 1]) and a color map ([0, 1] -> color)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Sphere
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        x = 0.9 * np.outer(np.cos(u), np.sin(v))
        y = 0.9 * np.outer(np.sin(u), np.sin(v))
        z = 0.9 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color=[0.5, 0.5, 0.5, 0.3])
        ax.plot_wireframe(x, y, z, rstride=2, cstride=2, color="k", lw=0.3)

        # Samples
        if X is not None:
            if type(X) == torch.Tensor:
                X = X.detach()

            _, _, gammas_X = self.identify_eulers_angles(X)
            ax.scatter(*X[:, :, 2].T, c=mappable.to_rgba(gammas_X), marker="o", depthshade=depthshade)

        _, _, gammas_Y = self.identify_eulers_angles(Y)
        for i in range(len(Y)):
            ax.scatter(*Y[i, :, :, 2].T, c=mappable.to_rgba(gammas_Y[i]), marker=f"${i}$", depthshade=depthshade)
        
        ax.set_box_aspect([1, 1, 1])
        plt.colorbar(mappable)
        plt.show()

    def plot_samples(self, Y, X=None, setlim=True, depthshade=True):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Samples
        if X is not None:
            if type(X) == torch.Tensor:
                X = X.detach()

            alphas_X, betas_X, gammas_X = self.identify_eulers_angles(X)
            ax.scatter(alphas_X, betas_X, gammas_X, depthshade=depthshade)

        alphas_Y, betas_Y, gammas_Y = self.identify_eulers_angles(Y)
        for i in range(len(Y)):
            ax.scatter(alphas_Y[i], betas_Y[i], gammas_Y[i], depthshade=depthshade)
        
        #ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("$\\alpha$")
        ax.set_ylabel("$\\beta$")
        ax.set_zlabel("$\\gamma$")
        if setlim:
            ax.set_xlim(0, 2*torch.pi)
            ax.set_ylim(0, torch.pi)
            ax.set_zlim(0, 2*torch.pi)
        plt.show()

class ScaSO3(SO3, ScaManifold):
    """Class computing barycentres on SO3 with the scalar product operator."""
    def __init__(self):
        super().__init__()


class DistSO3(SO3, SlicedManifold):
    """Class computing barycentres on SO3 with the SO3 distance operator."""
    def __init__(self, eps=1e-1):
        super().__init__()
        self.eps=eps
    
    def operator(self, psis, samples):
        """Projects the samples on the 1D axes corresponding to the different psis

        Args:
            psis (Tensor): Shape (n, *ms) where ms is the shape of the elements of the manifold
            samples (Tensor): Shape (*Ns, *ms) where Ns is a any shape, and prod(Ns) gives the number of samples
        
        Returns:
            projections (Tensor): Shape (*Ns, n)
        """
        sca_prod = torch.tensordot(samples, psis, dims=((self.dim_range), (self.dim_range))) # simple scalar product of every pair of vector
        dist_so3 = torch.arccos(torch.clip((sca_prod - 1)/2, -1, 1))
        return dist_so3
    
    def operator_gradient_factor(self, psis, samples):
        """Computes the gradient_factor of operator in every sample, for every psi. The gradient is then
        gardient_factor * psi.

        Args:
            psis (Tensor): Shape (n, *ms) where ms is the shape of the elements of the manifold
            samples (Tensor): Shape (*Ns, *ms) where Ns is any shape, and prod(Ns) gives the number of samples
        
        Returns:
            gradient_factors (Tensor): Shape (*Ns, n)
        """
        sca_prod = torch.tensordot(samples, psis, dims=((self.dim_range), (self.dim_range))) # simple scalar product of every pair of vector
        res = -1/torch.sqrt(4 - (sca_prod-1)**2 + self.eps)
        return res
