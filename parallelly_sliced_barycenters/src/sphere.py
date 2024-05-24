import torch
from scipy.stats import vonmises_fisher
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

from src.manifold import Manifold, ScaManifold, Measure, Mixture
from itertools import product as iter_prod

class Sphere(Manifold):
    """Class implementing the main functions for the sphere.
    It is designed for cooperative inheritance, along with a SWBarycenter computing barycentres.
    """
    def __init__(self, d):
        """
        Args:
            d (int): Dimension of the ambient space. The dimension of the sphere is thus d-1.
        """
        super().__init__()
        self.ms = (d,)
        self.d = d
    
    def dist(self, X0, X1):
        sca_prod = torch.sum(X0 * X1, dim=-1)
        return torch.arccos(sca_prod)
    
    def projection_tangent_space(self, X, dX):
        """Project a vector onto the tangential space
        
        Args:
            X (Tensor): Shape (*bs, d) or broadcastable to it. The points in which to consider the tangential spaces.
            dX (Tensor): Shape (*bs, d) or broadcastable to it. The vectors to project to the tangential spaces. 
        
        Returns:
            dX_p (Tensor): Shape (*bs, d). The vectors belonging to the tangential spaces.
        """
        normX2 = torch.sum(X**2, dim=self.dim_range, keepdim=True) # square norm of each X_k. Should be one.
        radial = torch.sum(X*dX, dim=self.dim_range, keepdim=True) / normX2 * X
        return dX - radial

    def exponential(self, X, dX):
        """Takes a vector from the tangential space to the sphere and returns the exponential map of it.

        Args:
            X (Tensor): Shape (*bs, d) or broadcastable to it. The points of reference on the sphere.
            dX (Tensor): Shape (*bs, d) or broadcastable to it. The tangential vectors, as produced by 
                self.projection_tangent_space.
        
        Returns:
            X_new (Tensor): Shape (*bs, d). The new points of the sphere corresponding to the 
                exponential of each dX in each X.
        """
        norm_dX = torch.sqrt(torch.sum(dX**2, dim=self.dim_range, keepdim=True))
        exp = torch.cos(norm_dX) * X + torch.sin(norm_dX) * dX / norm_dX
        return exp
    
    def logarithm(self, X0, X1):
        """Computes the/a logarithm of an element X1 relatively to element X0.

        Args:
            X0 (Tensor): Shape (*bs, d). The points of reference on the sphere.
            X1 (Tensor): Shape (*bs, d). The points to compute the/a logarithm of.

        Returns:
            Tensor: Shape (*bs, d). The vectors, in the tangent space of X0, corresponding to 
                the logarithm of X1 relatively to X0.
        """
        sca_prod = torch.sum(X0 * X1, dim=-1, keepdim=True)
        norm_dX = torch.arccos(sca_prod)
        dX0 = X1 - sca_prod * X0
        norm_dX0 = torch.norm(dX0, dim=-1, keepdim=True)

        for js_full in torch.nonzero(norm_dX0 < 1e-5): 
            # to handle particular case where norm_dX0 is null. We replace X1 by a random vector.
            js = js_full[:-1]
            random_vector = torch.randn(self.d)
            random_vector /= torch.norm(random_vector)
            dX0[*js] = random_vector -  torch.dot(random_vector, X0[*js]) * X0[*js]
            norm_dX0[*js] = torch.norm(dX0[*js])
        
        dX = dX0 / norm_dX0 * norm_dX
        return dX


    def projection_manifold(self, X):
        """Project a vector on the sphere.
        
        Args:
            X (Tensor): Shape (*bs, d). The vectors to project on the sphere.
        
        Returns:
            X_new (Tensor): Shape (*bs, d). The projected vectors.
        """
        return X / torch.sqrt(torch.sum(X**2, dim=self.dim_range, keepdim=True))
    
    def parametrisation(self, A):
        """Given spherical coordinates, returns a point on the sphere. More precisely, when spherical
            coordinates \phi_0 \in [0, 2\pi[, \phi_1 \in [0, \pi], ..., \phi_{d-2} \in [0, \pi] are 
            given, the d-dimensional coordinates are

                x_0 = sin(\phi_{d-2})...sin(\phi_1)sin(\phi_0)
                x_1 = sin(\phi_{d-2})...sin(\phi_1)cos(\phi_0)
                x_2 = sin(\phi_{d-2})...sin(\phi_2)cos(\phi_1)
                ...
                x_{d-2} = sin(\phi_{d-2})cos(\phi_{d-3})
                x_{d-1} = cos(\phi_{d-2})
        
        Args:
            A (Tensor): Shape (*bs, q) where q=d-1 is the dimension of the spherical coordinate space. 
                Coordinates of parametrised points.
        
        Returns:
            X (Tensor): Shape (*bs, d). Points on the sphere.
        """
        (*bs, _) = A.shape
        sin_angles = torch.sin(A)

        # sin_angles = torch.sqrt(1 - A**2) # first column is wrong
        # sin_angles[..., 0] = torch.sin(A)

        sin_angles_ext = torch.cat([sin_angles, torch.ones((*bs, 1))], dim=-1)
        X = torch.cumprod(sin_angles_ext.flip(-1), dim=-1).flip(-1) # cumulative prod in the other way
        X[..., 1:] *= torch.cos(A)

        # X[..., 2:] *= A[..., 1:]
        # X[..., 1] *= torch.cos(A[..., 0])
        return X
    
    def inverse_parametrisation(self, X):
        """Given points on the sphere, returns the coordinates of the points in the coordinate 
            space. It is the inverse function of self.parametrisation.

        Args:
            X (Tensor): Shape (*bs, d). Points on the sphere.

        Returns:
            A (Tensor): Shape (*bs, q) where q=d-1 is the dimension of the coordinate space. Coordinates of 
                the parametrised points.
        """
        y = torch.sqrt(torch.cumsum(X[..., :-1]**2, dim=-1))
        y[..., 0] = X[..., 0] # To retrieve the sign that was removed
        angles = torch.atan2(y, X[..., 1:]) # between -pi and pi 
        A = angles%(2*torch.pi)
        # A[..., 1:] = torch.cos(A[..., 1:])
        return A
    
    def sample_uniform(self, n):
        """Returns a sample from the uniform probability distribution on the sphere.
        
        Args:
            n (int): Number of samples.
        
        Returns:
            X (Tensor): Shape (n, d). Points on the sphere, uniformely distributed.
        """
        xhis = torch.randn(n, self.d)
        xhis = xhis/torch.linalg.norm(xhis, dim=1, keepdim=True)
        return xhis
    
    def sample_uniform_portion(self, bounds, n):
        """Not implemented"""
        raise NotImplementedError()
    
    def sample_vMF(self, coords, kappas, n):
        """Samples one or several von Mises - Fisher distributions on the (d-1)-sphere.

        Args:
            coords (Sequence): Shape (*bs, d). Mean directions of the VMF distributions.
            kappas (Sequence): Shape (*bs,). Concentration paramerters of the vMF distributions.
            n (int): Number of samples.

        Returns:
            Tensor: Shape (*bs, n, d). Sample points on the 2-sphere.
        """
        (*bs, _) = coords.shape
        if not type(kappas)== torch.Tensor:
            kappas = torch.tensor(kappas)
        kappas = torch.broadcast_to(kappas, bs)

        X = torch.zeros((*bs, n, self.d))
        for js in iter_prod(*[range(batch_dim_length) for batch_dim_length in bs]):
            X[*js] = torch.tensor(vonmises_fisher(coords[*js].cpu(), float(kappas[*js])).rvs(torch.tensor(n).cpu()))
        return X

    def plot_samples(self, Y, X, depthshade=True):
        """Not implemented"""
        raise NotImplementedError()


class Sphere2(Sphere):
    def __init__(self):
        super().__init__(d=3)
    
    def parametrisation(self, A):
        """Given spherical coordinates, returns a point on the 2-sphere. More precisely, when spherical
            coordinates \alpha \in [0, 2\pi[, \beta \in [0, \pi] are given, the 3-dimensional coordinates are

                x = sin(\beta)cos(\alpha)
                y = sin(\beta)sin(\alpha)
                z = cos(\beta).

            This is not the same as the parametrisation method of Sphere (of unspecified dimension), as axes x 
            and y are inverted, to keep usual conventions on S^2.
        
        Args:
            A (Tensor): Shape (*bs, q) where q=2 is the dimension of the spherical coordinate space. 
                Coordinates of parametrised points.
        
        Returns:
            X (Tensor): Shape (*bs, 3). Points on the sphere.
        """
        # alphas, cos_betas = A
        alphas, betas = A
        cos_betas = torch.cos(betas)
        sin_betas = torch.sin(betas)
        # sin_betas = torch.sqrt(1 - cos_betas**2)
        X = torch.zeros((*alphas.shape, 3))
        X[..., 0] = torch.cos(alphas)*sin_betas
        X[..., 1] = torch.sin(alphas)*sin_betas
        X[..., 2] = cos_betas
        return X

    def inverse_parametrisation(self, X):
        """Given points on the 2-sphere, returns the coordinates of the points in the coordinate 
            space. It is the inverse function of self.parametrisation. This is not the same as the 
            inverse_parametrisation method of Sphere (of unspecified dimension), as axes x and y 
            are inverted, to keep usual conventions on S^2.

        Args:
            X (Tensor): Shape (*bs, d). Points on the sphere.

        Returns:
            A (Tensor): Shape (*bs, q) where q=d-1 is the dimension of the coordinate space. Coordinates of 
                the parametrised points.
        """
        cos_betas = X[..., 2]
        alphas = torch.arctan(X[..., 1]/ X[..., 0])
        alphas += torch.pi * (X[..., 0] < 0)
        alphas = alphas % (2*torch.pi)
        betas = torch.arccos(cos_betas)
        # return alphas, cos_betas
        return alphas, betas
    
    def sample_uniform_portion(self, bounds, n):
        # Could be generalised to higher dimension if one knows how to sample (1-t**2)**((d-3)/2)dt
        """Returns a sample from the uniform probability distribution on a portion corresponding to a 
        parallelepipedeon in the spherical coordinate space.
        
        Args:
            bounds (Tensor): Shape (q, 2) where q=2 is the dimension of the spherical coordinate space. 
                The bounds (min, max) for each of the d coordinates.
            n (int): Number of samples.
        
        Returns:
            X (Tensor): Shape (n, 3). Points on the sphere uniformely distributed in the portion 
                described by bounds.
        """
        [a_min, a_max], [cb_min, cb_max] = bounds #TODO: Change input so that beta is not under cos form.
        alphas = torch.rand(n)*(a_max - a_min) + a_min
        # cb_min, cb_max = [torch.cos(b_max), torch.cos(b_min)]
        cos_betas = torch.rand(n)*(cb_max - cb_min) + cb_min
        alphas = torch.unsqueeze(alphas, -1)
        betas = torch.unsqueeze(torch.arccos(cos_betas), -1)
        return self.parametrisation(torch.cat([alphas, betas], -1))
        # return self.parametrisation(torch.stack([alphas, cos_betas]))

    def plot_samples(self, Y, X=None, depthshade=True, figname=None): 
        """Represent samples on the sphere.
        
        Args:
            Y (Tensor | list of tensor): Tensor of shape (M, Ny, 3) or list of length M of tensors of shape (Ny_j, 3).
                Target measures.
            X (Tensor): Shape (Nx, 3). Other empirical measure.
            depthshade (bool): Whether or not to change the color of the points according to their
                depth in the 3D plot space.
        """
        fig = plt.figure(figname)
        ax = fig.add_subplot(projection='3d')

        # Sphere
        u = torch.linspace(0, 2 * torch.pi, 60)
        v = torch.linspace(0, torch.pi, 60)
        x = 0.9 * torch.outer(torch.cos(u), torch.sin(v))
        y = 0.9 * torch.outer(torch.sin(u), torch.sin(v))
        z = 0.9 * torch.outer(torch.ones(u.size()), torch.cos(v))
        ax.plot_surface(x, y, z, color=[0.5, 0.5, 0.5, 0.3])
        ax.plot_wireframe(x, y, z, rstride=2, cstride=2, color="k", lw=0.3)

        # Samples
        if X is not None:
            if type(X) == torch.Tensor:
                X = X.detach()
            ax.scatter(*X.T, depthshade=depthshade)
        for i in range(len(Y)):
            ax.scatter(*Y[i].T, depthshade=depthshade)
        ax.set_box_aspect([1, 1, 1])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        # plt.show()
    
class Sphere3(Sphere):
    def __init__(self):
        super().__init__(d=4)
    
    def plot_samples(self, Y, X, depthshade=True, setlim=True, title=None, figname=None):
        """Represent samples on the 3-sphere, in the spherical coordinate space.
        
        Args:
            Y (Tensor | list of tensor): Tensor of shape (M, Ny, 3) or list of length M of tensors of shape (Ny_j, 3).
                Target measures.
            X (Tensor): Shape (Nx, 3). Other empirical measure.
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

            A_X = self.inverse_parametrisation(X)
            phiX_0, phiX_1, phiX_2 = torch.unbind(A_X, dim=-1)
            ax.scatter(phiX_0, phiX_1, phiX_2, depthshade=depthshade)

        A_Y = self.inverse_parametrisation(Y)
        phiY_0, phiY_1, phiY_2 = torch.unbind(A_Y, dim=-1)
        for i in range(len(Y)):
            ax.scatter(phiY_0[i], phiY_1[i], phiY_2[i], depthshade=depthshade)
        
        #ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("$\\phi_0$")
        ax.set_ylabel("$\\phi_1$")
        ax.set_zlabel("$\\phi_2$")
        if setlim:
            ax.set_xlim(0, 2*torch.pi)
            ax.set_ylim(0, torch.pi)
            ax.set_zlim(0, torch.pi)
        ax.set_title(title)
        plt.show()

    
class ScaSphere(Sphere, ScaManifold):
    """Class corresponding to the Sphere endowed with the scalar product. It can then compute Sliced
    Wasserstein barycentres."""
    def __init__(self, d):
        super().__init__(d=d)

class ScaSphere2(Sphere2, ScaManifold):
    """Class corresponding to the 2-Sphere endowed with the scalar product. It can then compute Sliced
    Wasserstein barycentres."""
    def __init__(self):
        super().__init__()

class ScaSphere3(Sphere3, ScaManifold):
    """Class corresponding to the 3-Sphere endowed with the scalar product. It can then compute Sliced
    Wasserstein barycentres."""
    def __init__(self):
        super().__init__()

#TODO: Find a nice way to make measures class live together with manifolds
class UniformSphere(Measure):
    def __init__(self, d=3):
        """
        Args:
            d (int, optional): Dimension of the ambient space. The dimension of the sphere is d-1. Defaults to 3.
        """
        self.d = d

    def sample(self, n):
        xhis = torch.randn(n, self.d)
        xhis = xhis/torch.linalg.norm(xhis, dim=1, keepdim=True)
        return xhis
    
    def pdf(self, points):
        volume = 2*torch.pi**(self.d/2) / gamma(self.d/2)
        return torch.ones((points.shape[0]))/volume #(4*torch.pi)
    

class UniformPortionSphere(Measure): #! Only available on the 2-sphere
    def __init__(self, bounds):
        """alpha, cos(beta)"""
        super().__init__()
        if type(bounds) == torch.Tensor:
            self.bounds = bounds.float()
        else:
            self.bounds = torch.tensor(bounds, dtype=torch.float)
    
    def sample(self, n):
        #TODO: Uniformise parametrisation (cos(beta) or beta)
        sphere = Sphere2()
        [a_min, a_max], [cb_min, cb_max] = self.bounds
        alphas = torch.rand(n)*(a_max - a_min) + a_min
        # cb_min, cb_max = [torch.cos(b_max), torch.cos(b_min)]
        cos_betas = torch.rand(n)*(cb_max - cb_min) + cb_min
        alphas = torch.unsqueeze(alphas, -1)
        betas = torch.unsqueeze(torch.arccos(cos_betas), -1)
        return sphere.parametrisation(torch.cat([alphas, betas], -1))
        # return sphere.parametrisation(torch.stack([alphas, cos_betas]))

class VMFSphere(Measure):
    def __init__(self, coords, kappa):
        super().__init__()
        if type(coords) == torch.Tensor:
            self.coords = coords.float()
        else:
            self.coords = torch.tensor(coords, dtype=torch.float)
        self.coords = self.coords/torch.norm(self.coords)
        self.kappa = float(kappa)

    def sample(self, n):
        return torch.tensor(vonmises_fisher(self.coords, self.kappa).rvs(n), dtype=torch.float)
    
    def pdf(self, points):
        A = torch.tensor(vonmises_fisher(self.coords, self.kappa).pdf(points), dtype=torch.float)
        return A

class SmileySphere(Mixture): #! Only avilable on the 2-sphere
    def __init__(self, cpt=[0., 0.9*torch.pi/2],  rcpt_eye = [0.4, -0.4], rcpt_mouth_ext = [0.6, 0.3], rct_mouth_middle = 0.5, kappa_eye=80, kappa_mouth=80, n_modes_mouth = 9, weight_eyes = 1/3):
        self.cpt = cpt # coordinates in terms of phi and theta
        self.rcpt_eye = rcpt_eye #  relative cpt of the right eye
        self.rcpt_mouth_ext = rcpt_mouth_ext # relative cpt of the right extremity of the mouth
        self.rct_mouth_middle = rct_mouth_middle # relative coordinate in terms of theta of the middle of the mouth. phi = 0
        self.kappa_eye = kappa_eye
        self.kappa_mouth = kappa_mouth
        self.n_modes_mouth = n_modes_mouth

        coords = self.compute_coords()
        kappas = [kappa_eye, kappa_eye] + [kappa_mouth] * n_modes_mouth
        measures = [VMFSphere(coord, kappa) for coord, kappa in zip(coords, kappas)]
        lambdas = [weight_eyes/2] * 2 + [(1 - weight_eyes)/n_modes_mouth] * n_modes_mouth
        lambdas = torch.tensor(lambdas)
        super().__init__(*measures, lambdas=lambdas) 

    def compute_coords(self):
        t = torch.linspace(-1, 1, self.n_modes_mouth)
        phi_ext, theta_ext = self.rcpt_mouth_ext
        theta_middle = self.rct_mouth_middle
        ph_mouth = t * phi_ext
        th_mouth = (theta_ext - theta_middle) * t**2 + theta_middle

        phi_eye, theta_eye = self.rcpt_eye
        ph = torch.cat((torch.tensor([ phi_eye , -phi_eye ]), ph_mouth))
        th = torch.cat((torch.tensor([theta_eye, theta_eye]), th_mouth))
        cpt_measures = torch.stack([ph, th]) + torch.tensor(self.cpt)[:, None]
        # cpt_measures[1] = torch.cos(cpt_measures[1])
        cpt_measures = cpt_measures.T

        sphere = Sphere2()
        coords = sphere.parametrisation(cpt_measures)
        return coords

class SphereDiscretisation(): #! Only available on the 2-sphere
    """Builds a grid on the sphere and plots functions defined on this grid.
    Points of the grid are ~ uniformly distributed."""
    def __init__(self, N1, N2):
        """
        Args:
            N1 (int): Number of meridians (i.e. points along the equator).
            N2 (int): Number of points along the z axis.
        """
        self.N1 = N1
        self.N2 = N2
        self.xb = None
        self.yb = None
        self.zb = None
        self.pb = None # phi, bounds
        self.hb = None # height (z), bounds
        self.pp = None # phi, middle point
        self.hp = None # height, middle poin
        self.convert = None
        self.support = None

    def angles_to_coords(self, u, v):
        return torch.outer(torch.cos(u), torch.sqrt(1-v**2)), torch.outer(torch.sin(u), torch.sqrt(1-v**2)), torch.outer(torch.ones(u.size()), v)         

    def build_mesh(self):
        """Builds a grid on the sphere. Points of the grid are ~ uniformly distributed."""
        pi = torch.pi

        # Borders of the rectangles
        self.pb = torch.linspace(-pi, pi, self.N1 + 1)
        self.hb = torch.linspace(-1, 1, self.N2 + 1)
        self.xb, self.yb, self.zb = self.angles_to_coords(self.pb, self.hb)

        # Points
        self.pp = (self.pb[:-1] + self.pb[1:])/2
        self.hp = (self.hb[:-1] + self.hb[1:])/2
        xp, yp, zp = self.angles_to_coords(self.pp, self.hp)

        self.convert = torch.arange(self.N1*self.N2).reshape(self.N1, self.N2)
        A, B = torch.meshgrid(torch.arange(self.N1), torch.arange(self.N2), indexing="ij")
        convert_inv = torch.stack([A, B], dim=2)
        self.convert_inv = convert_inv.reshape(-1, 2)
        self.support = torch.stack([xp, yp, zp], dim=2)[self.convert_inv[:,0], self.convert_inv[:,1]]
        return self.support
    
    def plot_colours(self, measures, barycenter, projection="mollweide", figname=None): 
        M = len(self.measures)
        colours = torch.rand((M+1, 3))
        colours /= torch.max(colours, dim=1, keepdim=True).values
        colours[:2] = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float) # 2 first measures
        colours[-1] = torch.tensor([0, 0, 1], dtype=torch.float) # barycenter

        VW = torch.cat([measures, barycenter.unsqueeze(0)], dim=0) # (M+1, N)
        VW /= torch.max(VW, dim=1, keepdim=True).values
        # blend = VW.unsqueeze(0).repeat((M + 1, 1, 1)) # (M+1, M+1, N)
        # eye = torch.eye(M+1).unsqueeze(-1)            # (M+1, M+1)
        # blend = eye*blend + (1-eye)*(1-blend)         # (M+1, M+1, N)
        # colour_weights = torch.prod(blend, dim=1)     # (M+1, N)
        # final_colours = colour_weights.T @ colours    # (N, 3)

        # final_colours = (VW.T**2) @ colours / VW.T.sum(dim=1, keepdim=True)
        final_colours = VW.T @ colours # (N, 3)
        final_colours = final_colours[self.convert] # (N1, N2, 3)

        # alphas = torch.arctan(self.support[:, 1]/self.support[:,0])
        # alphas += torch.pi*(self.support[:,0] < 0)
        # alphas = (alphas + torch.pi)%(2*torch.pi) - torch.pi
        # plt.scatter(alphas, self.support[:, 2], c=final_colours)
        # plt.title(title)
        # plt.show()

        if projection == "3d":
            fig = plt.figure(figname)
            ax = fig.add_subplot(projection='3d')
            surf = ax.plot_surface(self.xb, self.yb, self.zb, facecolors=final_colours, rstride=1, cstride=1)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            plt.show()
        else:
            fig = plt.figure(figname)
            ax = fig.add_subplot(projection=projection)
            pb, lb = torch.meshgrid(self.pb, torch.arcsin(self.hb), indexing="ij") # phi (longitude), lambda (latitude)
            pm = ax.pcolormesh(pb, lb, final_colours)
            ax.grid(True)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            fig.colorbar(pm, ax=ax, shrink=0.6)
            plt.show()
    
    def plot_wire(self, measures, barycentre, figname=None):
        # (M, N), (N)
        measures_conv = measures[:, self.convert]
        barycentre_conv = barycentre[self.convert]
        
        fig = plt.figure(figname)
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(self.xb, self.yb, self.zb, color=[0.5, 0.5, 0.5, 0.3])
        f = 1 + self.N1*self.N2/20*barycentre_conv
        ax.plot_wireframe(f*self.support[:, 0], f*self.support[:, 1], f*self.support[:, 2], rcount=70, ccount=30, lw=0.3)
        for i, m in enumerate(measures_conv):
            f = 1 + self.N1*self.N2/20*m
            ax.plot_wireframe(f*self.support[:, 0], f*self.support[:, 1], f*self.support[:, 2], rcount=70, ccount=30, color=f"C{i+1}", lw=0.3)
        ax.set_box_aspect([1, 1, 1])
        plt.show()

    def plot(self, measure, projection="3d", contour=False, figname=None):
        # measure : Shape (N)
        measure_conv = measure[self.convert] # (N1, N2)
        # measure_conv = measure_conv / measure_conv.max()

        my_colors = torch.Tensor([
            [ 63,  40, 174],
            [ 51, 122, 253],
            [ 16, 178, 213],
            [ 99, 205, 112],
            [247, 186,  61],
            [248, 248,  23],
        ]).numpy()/256 # colormap in Michael paper
        cm = mpl.colors.LinearSegmentedColormap.from_list("my_cm", my_colors)

        if projection == "3d":
            # cm = plt.colormaps["viridis"]
            norm = mpl.colors.Normalize(0, measure_conv.max())
            scalar_mappable = mpl.cm.ScalarMappable(norm = norm, cmap = cm)
            
            fig = plt.figure(figname)
            ax = fig.add_subplot(projection='3d')
            surf = ax.plot_surface(self.xb, self.yb, self.zb, facecolors=cm(norm(measure_conv)), rstride=1, cstride=1)
            fig.colorbar(scalar_mappable, ax=ax)
            ax.set_box_aspect([1, 1, 1])
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_zlabel("z")
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            ax.view_init(azim=45)
            plt.show()
        elif not contour:
            fig = plt.figure(figname)
            ax = fig.add_subplot(projection=projection)
            pb, lb = torch.meshgrid(self.pb, torch.arcsin(self.hb), indexing="ij") # phi (longitude), lambda (latitude)
            pm = ax.pcolormesh(pb, lb, measure_conv)
            ax.grid(True)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            fig.colorbar(pm, ax=ax, shrink=0.6)
            plt.show()
        else :
            fig = plt.figure(figname)
            ax = fig.add_subplot(projection=projection)
            levels = MaxNLocator(nbins=10).tick_values(measure_conv.min(), measure_conv.max())
            cf = ax.contourf(self.pp.T, torch.arcsin(self.hp).T, measure_conv.T, levels=levels)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            fig.colorbar(cf, ax=ax, shrink=0.4)
            plt.show()

    