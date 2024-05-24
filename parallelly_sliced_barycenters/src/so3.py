import torch
import numpy as np
from scipy.stats import vonmises_fisher
from scipy.spatial.transform import Rotation
from scipy.linalg import logm

from src.manifold import Manifold, SlicedManifold, ScaManifold

from itertools import product as iter_prod
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

class SO3(Manifold):
    """Class implementing the main functions for the SO3 manifold.
    It is designed for cooperative inheritance, along with a class using those operations."""
    def __init__(self):
        super().__init__()
        self.ms = (3, 3)
    
    def dist(self, X0, X1):
        sca_prod = torch.sum(X0 * X1, dim=(-2, -1))
        return torch.arccos((sca_prod - 1) / 2)

    def projection_tangent_space(self, X, dX):
        """Projects a vector onto the tangential space.
        
        Args:
            X (Tensor): Shape (*bs, 3, 3) or broadcastable to it. The points in which to consider the tangential spaces.
            dX (Tensor): Shape (*bs, 3, 3) or broadcastable to it. The vectors to project to the tangential spaces. 
        
        Returns:
            dX_p (Tensor): Shape (*bs, 3, 3). The vectors belonging to the tangential spaces.
        """
        PdX = 1/2 * (dX - X @ torch.transpose(dX, -2, -1) @ X)
        return PdX

    def cayley_retraction(self, X, dX):
        """Cayley retraction of dX (tangent vector in X for SO3) in X on SO3.

        Args:
            X (Tensor): Shape (*bs, 3, 3). Rotation matrices.
            dX (Tensor): Shape (*bs, 3, 3). Tangent vectors of SO3 in the points given by X.

        Returns:
            (Tensor): Shape (*bs, 3, 3). Each is the cay retraction on SO3.
        """
        # W = X.transpose(0, 2, 1) @ dX
        CayW = torch.linalg.inv(X - 1/2 * dX) @ (X + 1/2 * dX)
        return X @ CayW

    def exponential(self, X, dX):
        """Takes a vector from the tangential space to SO(3).

        Args:
            X (Tensor): Shape (*bs, 3, 3). The points of reference on SO(3).
            dX (Tensor): Shape (*bs, 3, 3). The tangential vectors, as produced by self.projection_tangent_space.
        
        Returns:
            X_new (Tensor): Shape (*bs, 3, 3). The new points of SO(3) corresponding to the 
                exponential of each dX in each X.
        """
        return X @ torch.linalg.matrix_exp(torch.transpose(X, -2, -1) @ dX)
    
    def logarithm(self, X0, X1, method=1):
        """Computes the/a logarithm of an element X1 relatively to element X0.

        Args:
            X0 (Tensor): Shape (*bs, 3, 3). The points of reference on SO(3).
            X1 (Tensor): Shape (*bs, 3, 3). The points to compute the/a logarithm of.

        Returns:
            Tensor: Shape (*bs, 3, 3). The vectors, in the tangent space of X0, corresponding to 
                the logarithm map of X1 relatively to X0.
        """
        R = torch.transpose(X0, -2, -1) @ X1 # rotation between X0 and X1

        # computation of log(R)
        if method == 1:
            # Method 1. Does not work when -1 is an eigenvalue of R.
            diags = R[..., [0, 1, 2], [0, 1, 2]] # diagonal of each rotation matrix. (*bs, 3)
            trace = torch.sum(diags, -1)
            thetas = torch.zeros(R.shape)
            thetas[...] = torch.arccos((trace - 1)/2)[..., None, None] #(*bs, 1, 1)
            log_R = thetas / (2 * torch.sin(thetas)) * (R - torch.transpose(R, -2, -1))
            log_R[thetas == 0] = 0
        else:
            # Method 2
            log_R = torch.zeros(R.shape)
            bs = R.shape[:-2]
            for js in iter_prod(*[range(N) for N in bs]):
                log_Rj, error = logm(R[*js], disp = False)
                log_R[*js] = torch.tensor(log_Rj)
        
        return X0 @ log_R
        
    
    def projection_manifold(self, X):
        """Projects a vector on SO(3).
        
        Args:
            X (Tensor): Shape (*bs, 3, 3). The vectors to project on SO(3).
        
        Returns:
            X_new (Tensor): Shape (*bs, 3, 3). The projected vectors.
        """
        U, S, Vh = torch.linalg.svd(X) #SVD of each vector in X
        return U @ Vh #matmul of each (U[i_1, ..., i_k], Vh[i_1, ..., i_k]) where k=len(Xs)
    
    def axis_angle_to_matrix(self, axes, angles):
        """Generates rotation matrices corresponding to the given axes and teh given angles.

        Args:
            axes (Tensor): Rotation axes. Shape (*bs, 3).
            angles (Tensor): Angles. Shape (*bs,).

        Returns:
            (Tensor): Shape (*bs, 3, 3), rotation matrices.
        """
        c = torch.cos(angles)
        s = torch.sin(angles)
        bs = angles.shape
        batch_ndim = len(bs)

        axes = torch.permute(axes, (-1, *range(batch_ndim))) # (3, *bs)

        t1 = axes[:,None]*axes[None,:] * (1-c) # (3, 3, *bs)
        t2 = torch.stack([torch.stack([    c     , -axes[2]*s,  axes[1]*s]),
                          torch.stack([ axes[2]*s,     c     , -axes[0]*s]),
                          torch.stack([-axes[1]*s,  axes[0]*s,     c     ])]) # (3, 3, *bs)
        t = t1 + t2
        t = torch.permute(t, (*range(-batch_ndim, 0), 0, 1))
        return t

    def cano_angle_to_matrix(self, axis, angles):
        """Generates rotation matrices corresponding to the given angles, for a given fixed canonical axis.

        Args:
            axis (int): 0, 1, 2. The rotation axis.
            angles (torch.Tensor | float): Angles. Shape (*bs,).

        Returns:
            torch.Tensor(*bs, 3, 3): Rotation matrices.
        """
        if type(angles) != torch.Tensor:
            angles = torch.tensor(angles)
        
        c = torch.cos(angles)
        s = torch.sin(angles)
        bs = angles.shape
        batch_ndim = len(bs)

        t = torch.zeros((3, 3, *bs))
        t[0, 0] = 1
        t[[1, 2], [1, 2]] = c
        t[2, 1] = s
        t[1, 2] = -s

        # one = torch.ones(bs)
        # zer = torch.zeros(bs)
        # t = torch.tensor([[one, zer, zer], [zer, c, -s], [zer, s, c]]) # (3, 3, *bs)

        perm = (torch.arange(3) - axis) % 3
        t = t[perm[:, None], perm[None, :]]

        t = torch.permute(t, (*range(-batch_ndim, 0), 0, 1)) # (*bs, 3, 3)
        return t

    def parametrisation(self, A):
        #TODO: Change input so that the batch size comes first, to be coherent with sphere
        """Returns a point on SO(3) given Euler angles.
        
        Args:
            A (Tensor): Shape (q, *bs) where q=3 is the dimension of the coordinate space of Euler angles.
                Coordinates (alpha, beta, gamma) of parametrised points R_3(alpha)R_2(beta)R_3(gamma), 
                with alpha, gamma \in [0, 2\pi[ and beta \in [0, \pi].
        
        Returns:
            X (Tensor): Shape (*bs, 3, 3). Points on SO(3).
        """
        alphas, betas, gammas = A
        r3a = self.cano_angle_to_matrix(2, alphas)
        r2b = self.cano_angle_to_matrix(1, betas)
        r3g = self.cano_angle_to_matrix(2, gammas)
        return r3a @ r2b @ r3g
    
    def inverse_parametrisation(self, rotations):
        """Given matrices in SO(3), returns the Euler angles of them. It is the inverse function of 
            self.parametrisation.

        Args:
            X (Tensor): Shape (*bs, 3, 3). Points on the SO(3).

        Returns:
            A (Tensor): Shape (q, *bs) where q=3 is the dimension of the coordinate space of Euler angles. 
                Coordinates (alpha, beta, gamma) of parametrised points R_3(alpha)R_2(beta)R_3(gamma), 
                with alpha, gamma \in [0, 2\pi[ and beta \in [0, \pi].
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
    
    def sample_uniform(self, n):
        """Returns a sample from the uniform probability distribution on SO(3).
        
        Args:
            n (int): Number of samples.
        
        Returns:
            X (Tensor): Shape (n, 3, 3). Points on SO(3), uniformely distributed.
        """
        # xhis = np.random.randn(n, 3, 3)
        # Qs, _ = np.linalg.qr(xhis)
        # # It seems that the determinant is always +1
        # return Qs

        alphas = torch.rand(n)* 2 * torch.pi
        betas  = torch.arccos(torch.rand(n)* 2 - 1)
        gammas = torch.rand(n)* 2 * torch.pi
        A = torch.stack([alphas, betas, gammas])
        #TODO: Change so that the batch size comes first, to be coherent with sphere
        return self.parametrisation(A)
    
    def sample_uniform_portion(self, bounds, n):
        """Returns a sample from the uniform probability distribution on a portion corresponding to a 
        parallelepipedeon in the coordinate space of Euler angles.
        
        Args:
            bounds (Tensor): Shape (q, 2) where q=3 is the dimension of the coordinate space. 
                The bounds (min, max) for each of the q coordinates.
            n (int): Number of samples.
        
        Returns:
            X (Tensor): Shape (n, 3, 3). Points on SO(3) uniformely distributed in the portion 
                described by bounds.
        """
        a_min, a_max = bounds[0]
        cb_max, cb_min = torch.cos(bounds[1])
        g_min, g_max = bounds[2]

        alphas = torch.rand(n)*(a_max - a_min) + a_min
        betas  = torch.arccos(torch.rand(n)*(cb_max - cb_min) + cb_min)
        gammas = torch.rand(n)*(g_max - g_min) + g_min

        A = torch.stack([alphas, betas, gammas])
        #TODO: Change so that batch size comes first, to be coherent with sphere
        return self.parametrisation(A)
    
    def sample_vMF_quat(self, coords, kappas, n):
        """Samples one or several distributions on SO(3) corresponding to von-Mises-Fisher distributions\
        on the unit sphere of quaternions.

        Args:
            coords (Tensor | ndarray): Shape (*bs, 3, 3). Mean directions of the distributions, expressed as rotation matrices.
            kappas (Sequence | float): Shape (*bs,). Concentration parameters of the distributions.
            n (int): Number of samples in each distribution.

        Returns:
            Tensor: Shape (*bs, 3, 3). Samples, expressed as rotations matrices.
        """
        (*bs, _, _) = coords.shape
        if not type(kappas)== torch.Tensor:
            kappas = torch.tensor(kappas)
        kappas = torch.broadcast_to(kappas, bs)
        
        X = torch.zeros((*bs, n, 3, 3))
        for js in iter_prod(*[range(m) for m in bs]):
            mu_quat = Rotation.from_matrix(coords[*js]).as_quat()
            samples_quat = torch.tensor(vonmises_fisher(mu_quat, float(kappas[*js])).rvs(n))
            X[*js] = torch.tensor(Rotation.from_quat(samples_quat).as_matrix())
        return X
    
    def plot_samples_sphere(self, Y, X=None, depthshade=True):
        """Represents samples on SO(3). Points are represented on a sphere. Their coordinates on the 
            sphere represents alpha and beta, and their colour represents gamma.
        
        Args:
            Y (Tensor): Shape (M, N, 3, 3). Target measures.
            X (Tensor): Shape (N, 3, 3). Other empirical measure.
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

            _, _, gammas_X = self.inverse_parametrisation(X)
            ax.scatter(*X[:, :, 2].T, c=mappable.to_rgba(gammas_X), marker="o", depthshade=depthshade)

        _, _, gammas_Y = self.inverse_parametrisation(Y)
        for i in range(len(Y)):
            ax.scatter(*Y[i, :, :, 2].T, c=mappable.to_rgba(gammas_Y[i]), marker=f"${i}$", depthshade=depthshade)
        
        ax.set_box_aspect([1, 1, 1])
        plt.colorbar(mappable)
        plt.show()

    def plot_samples_euler(self, Y, X=None, setlim=True, depthshade=True, title=None, figname=None):
        """Represents samples on SO(3). Points are represented in the 3D space of Euler angles.
        
        Args:
            Y (Tensor): Shape (M, N, 3, 3). Target measures.
            X (Tensor): Shape (N, 3, 3). Other empirical measure.
            depthshade (bool): Whether or not to change the color of the points according to their
                depth in the 3D plot space.
            setlim (bool): Whether to display the whole coordinate space (True) or to let matplotlib
                zoom on the points (False).
            title (str | None): Title of the axis.
            figname (str | None): Name of the figure.
        """
        fig = plt.figure(figname)
        ax = fig.add_subplot(projection='3d')

        # Samples
        if X is not None:
            if type(X) == torch.Tensor:
                X = X.detach()

            alphas_X, betas_X, gammas_X = self.inverse_parametrisation(X)
            ax.scatter(alphas_X, betas_X, gammas_X, depthshade=depthshade)

        alphas_Y, betas_Y, gammas_Y = self.inverse_parametrisation(Y)
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
        ax.set_title(title)
        plt.show()
    
    def plot_samples_axis_angle(self, Y, X=None, depthshade=True, title=None, figname=None):
        """Represents samples on SO(3). Points of SO(3) are represented by the stereographic projection of 
            the corresponding quaternions.

            \pi(R) := \tan(\alpha/4) r = (q2, q3, q4) / (1 + q1)

            where r is the axis of rotation of the rotation R, \alpha \in [0, \pi] is its angle of rotation,
            and q=(q1, q2, q3, q4), with q1 > 0, is one of the corresponding quaternions.
        
        Args:
            Y (Tensor): Shape (M, N, 3, 3). Target measures.
            X (Tensor): Shape (N, 3, 3). Other empirical measure.
            depthshade (bool): Whether or not to change the color of the points according to their
                depth in the 3D plot space.
            title (str | None): Title of the axis.
            figname (str | None): Name of the figure.
        """
        fig = plt.figure(figname)
        ax = fig.add_subplot(projection='3d')

        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color=[0.5, 0.5, 0.5, 0.1])
        ax.view_init(elev=90.)

        # Samples
        if X is not None:
            if type(X) == torch.Tensor:
                X = X.detach()

            quats_X = Rotation.from_matrix(X).as_quat()  #scalar component is at the end
            quats_X = np.where((quats_X[:, 3] >= 0)[:, None], quats_X, -quats_X)
            stereo_proj_X = quats_X[:, :3] / (1 + quats_X[:, [3]])
            ax.scatter(stereo_proj_X[:, 0], stereo_proj_X[:, 1], stereo_proj_X[:, 2], depthshade=depthshade)

        for j in range(len(Y)):
            quats_Yj = Rotation.from_matrix(Y[j]).as_quat()
            quats_Yj = np.where((quats_Yj[:, 3] >= 0)[:, None], quats_Yj, -quats_Yj)
            stereo_proj_Yj = quats_Yj[:, :3]/ (1+ quats_Yj[:, [3]])
            ax.scatter(stereo_proj_Yj[:, 0], stereo_proj_Yj[:, 1], stereo_proj_Yj[:, 2], depthshade=depthshade)
        
        ax.set_box_aspect([1, 1, 1])
        # ax.set_xlabel(blabla)
        # ax.set_xlim(0, 2*torch.pi)
        
        ax.set_title(title)
        plt.axis('off')
        if figname:
            plt.savefig(figname + ".png")
        else:
            plt.show()
    
    def plot_samples(self, Y, X=None, depthshade=True, method="axis angle", **kwargs):
        """Represents samples on SO(3). 
        
        Args:
            Y (Tensor): Shape (M, N, 3, 3). Target measures.
            X (Tensor): Shape (N, 3, 3). Other empirical measure.
            depthshade (bool): Whether or not to change the color of the points according to their
                depth in the 3D plot space.
            method (str): How to map samples to R^3. Available options are "sphere", "euler", "axis angle".
                Defaults to "axis angle".
        """
        if method == "sphere":
            return self.plot_samples_sphere(Y, X, depthshade=depthshade, **kwargs)
        elif method == "euler":
            return self.plot_samples_euler(Y, X, depthshade=depthshade, **kwargs)
        elif method == "axis angle":
            return self.plot_samples_axis_angle(Y, X, depthshade=depthshade, **kwargs)
        else:
            raise ValueError(f"'method' argument should be in ['sphere', 'euler', 'axis angle']. {method} given.")

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
        """Projects the samples on the 1D axes corresponding to the different psis.

        Args:
            psis (Tensor): Shape (n, 3, 3). The directions of the slices.
            samples (Tensor): Shape (*bs, 3, 3) where bs is the batch size.
        
        Returns:
            projections (Tensor): Shape (*bs, n).
        """
        sca_prod = torch.tensordot(samples, psis, dims=((self.dim_range), (self.dim_range))) # simple scalar product of every pair of vector
        dist_so3 = torch.arccos(torch.clip((sca_prod - 1)/2, -1, 1))
        return dist_so3
    
    def operator_gradient_factor(self, psis, samples):
        """Computes the gradient_factor of operator in every sample, for every psi. The gradient is then
        gardient_factor * psi.

        Args:
            psis (Tensor): Shape (n, 3, 3). The directions of the slices.
            samples (Tensor): Shape (*bs, 3, 3) where bs is the batch size.
        
        Returns:
            gradient_factors (Tensor): Shape (*bs, n).
        """
        sca_prod = torch.tensordot(samples, psis, dims=((self.dim_range), (self.dim_range))) # simple scalar product of every pair of vector
        res = -1/torch.sqrt(4 - (sca_prod-1)**2 + self.eps)
        return res
