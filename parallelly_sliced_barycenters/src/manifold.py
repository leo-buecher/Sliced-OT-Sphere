import torch
from utils import unsqueeze

from abc import ABC, abstractmethod

class Manifold(ABC):
    def __init__(self):
        self.ms=None
    
    @property
    def ndim(self):
        return len(self.ms)
    
    @property
    def dim_range(self):
        return tuple(i for i in range(-self.ndim, 0))

    @abstractmethod
    def projection_tangent_space(self, X, dX):
        """Projects a vector onto the tangential space.
        
        Args:
            X (Tensor): Shape (*bs, *ms). The points in which to consider the tangential spaces.
            dX (Tensor): Shape (*bs, *ms). The vectors to project to the tangential spaces. 
        
        Returns:
            dX_p (Tensor): Shape (*bs, *ms). The vectors belonging to the tangential spaces.
        """
        pass

    @abstractmethod
    def exponential(self, X, dX):
        """Takes a vector from the tangential space to the manifold.

        Args:
            X (Tensor): Shape (*bs, *ms). The points of reference on the manifold.
            dX (Tensor): Shape (*bs, *ms). The tangential vectors, as produced by self.projection_tangent_space.
        
        Returns:
            X_new (Tensor): Shape (*bs, *ms). The new points of the manifold corresponding to the 
                exponential of each dX in each X.
        """
        pass

    @abstractmethod
    def logarithm(self, X0, X1):
        """Computes the/a logarithm of an element X1 relatively to element X0.

        Args:
            X0 (Tensor): Shape (*bs, *ms). The points of reference on the manifold.
            X1 (Tensor): Shape (*bs, *ms). The points to compute the/a logarithm of.

        Returns:
            Tensor: Shape (*bs, *ms). The vectors, in the tangent space of X0, corresponding to 
                the logarithm map of X1 relatively to X0.
        """
        pass
    
    @abstractmethod
    def projection_manifold(self, X):
        """Projects a vector on the manifold.
        
        Args:
            X (Tensor): Shape (*bs, *ms). The vectors to project on the manifold.
        
        Returns:
            X_new (Tensor): Shape (*bs, *ms). The projected vectors.
        """
        pass
    
    @abstractmethod
    def parametrisation(self, A):
        """Returns a point on the manifold from a point in R^q.
        
        Args:
            A (Tensor): Shape (*bs, q) where q is the dimension of the coordinate space. Coordinates of
                parametrised points.
        
        Returns:
            X (Tensor): Shape (*bs, *ms). Points on the manifold.
        """
        pass

    @abstractmethod
    def inverse_parametrisation(self, X):
        """Given points on the manifolds, returns the coordinates of the points in the coordinate 
            space. It is the inverse function of self.parametrisation.

        Args:
            X (Tensor): Shape (*bs, *ms). Points on the manifold.

        Returns:
            A (Tensor): Shape (*bs, q) where q is the dimension of the coordinate space. Coordinates of 
                the parametrised points.
        """
    
    @abstractmethod
    def sample_uniform(self, n):
        """Returns a sample from the uniform probability distribution on the manifold.
        
        Args:
            n (int): Number of samples.
        
        Returns:
            X (Tensor): Shape (n, *ms). Points on the manifold, uniformely distributed.
        """
        pass
    
    @abstractmethod
    def sample_uniform_portion(self, bounds, n):
        """Returns a sample from the uniform probability distribution on a portion corresponding to a 
        parallelepipedeon in the coordinate space.
        
        Args:
            bounds (Tensor): Shape (q, 2) where q is the dimension of the coordinate space. 
                The bounds (min, max) for each of the q coordinates.
            n (int): Number of samples.
        
        Returns:
            X (Tensor): Shape (n, *ms). Points on the manifold uniformely distributed in the portion 
                described by bounds.
        """
        pass
    
    def sample_projected_gaussian(self, coords, stds, N):
        """Generates one or several samples of gaussians projected on the manifold.

        Args:
            coords (Tensor, np.ndarray, list): Shape (*bs, *ms). Coordinates of the center(s) of the gaussian(s).
            stds (Tensor, np.ndarray, list, int): Shape (*s) such that bs=(..., *s).
                Standard deviations of the different gaussians.
            N (int): Number of samples in each set.

        Returns:
            Tensor: Shape (*bs, N, *ms)
        """
        if type(coords) != torch.Tensor:
            coords = torch.tensor(coords, dtype=torch.float)
        # M = coords.shape[0]
        bs = coords.shape[:-self.ndim]
        coords = unsqueeze(coords, len(bs), 1)

        if type(stds) != torch.Tensor:
            stds = torch.tensor(stds, dtype=torch.float)
        stds = unsqueeze(stds, 0, len(bs) - stds.ndim) # broadcastable to (*bs,)
        stds = unsqueeze(stds, -1, 1 + self.ndim) # broadcastable to (*bs, N, *ms)

        Y = (torch.randn(*bs, N, *self.ms) * stds) + coords
        Y = self.projection_manifold(Y)
        return Y
    
    @abstractmethod
    def plot_samples(self, Y, X, depthshade=True):
        """Represents samples on the manifold.
        
        Args:
            Y (Tensor): Shape (M, N, *ms). Target measures.
            X (Tensor): Shape (N, *ms). Other empirical measure.
            depthshade (bool): Whether or not to change the color of the points according to their
                depth in the 3D plot space.
        """
        pass

class SlicedManifold(Manifold):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def operator(self, psis, samples):
        """Projects the samples on the 1D axes corresponding to the different psis.

        Args:
            psis (Tensor): Shape (n, *ms) where ms is the shape of the elements of the manifold.
            samples (Tensor): Shape (*bs, *ms) where bs is the batch size.
        
        Returns:
            projections (Tensor): Shape (*bs, n).
        """
        pass
    
    @abstractmethod
    def operator_gradient_factor(self, psis, samples):
        """Computes the gradient_factor of operator in every sample, for every psi. The gradient is then
        gardient_factor * psi.

        Args:
            psis (Tensor): Shape (n, *ms) where ms is the shape of the elements of the manifold.
            samples (Tensor): Shape (*bs, *ms) where bs is the batch size.
        
        Returns:
            gradient_factors (Tensor): Shape (*bs, n) or broadcastable to it.
        """
        pass

class ScaManifold(SlicedManifold):
    """Class computing Sliced Wasserstein Barycentres, using the scalar product as operator 
    projecting the measures to the real line and can compute the stochastic gradient descent loop 
    giving the corresponding sliced Wasserstein barycentre.

    It is designed to be used in cooperative inheritance, along with a specific manifold.
    """
    def __init__(self):
        super().__init__()
    
    def operator(self, psis, samples):
        """Projects the samples on the 1D axes corresponding to the different psis.

        Args:
            psis (Tensor): Shape (n, *ms) where ms is the shape of the elements of the manifold.
            samples (Tensor): Shape (*bs, *ms) where bs is the batch size.
        
        Returns:
            projections (Tensor): Shape (*bs, n).
        """
        return torch.tensordot(samples, psis, dims=((self.dim_range), (self.dim_range))) # simple scalar product of every pair of vector
    
    def operator_gradient_factor(self, psis, samples):
        """Computes the gradient_factor of operator in every sample, for every psi. The gradient is then
        gardient_factor * psi.

        Args:
            psis (Tensor): Shape (n, *ms) where ms is the shape of the elements of the manifold.
            samples (Tensor): Shape (*bs, *ms) where bs is the batch size.
        
        Returns:
            gradient_factors (Tensor): Shape (*bs, n) or broadcastable to it. In this sub class,
                returns number 1.
        """
        return 1


class Measure():
    def __init__(self):
        pass

    def sample(self, n):
        raise NotImplementedError
    
    def pdf(self, points):
        raise NotImplementedError
    
    def discretise_on(self, points):
        """Supposes the points are uniformly distributed on the sphere."""
        f = self.pdf(points)
        return f/f.sum()
    
class Mixture(Measure):
    def __init__(self, *measures, lambdas=None):
        self.measures = measures
        M = len(self.measures)
        self.lambdas = lambdas if lambdas is not None else 1/M * torch.ones((M,))
    
    def sample(self, N):
        Ns = torch.distributions.Multinomial(N, self.lambdas).sample().long() # divide N into n integers (where n=len(lambdas)), according to the weights lambdas.
        X = []
        for i in range(len(self.measures)):
            X.append(self.measures[i].sample(Ns[i]))
        X = torch.concatenate(X)
        X = X[torch.randperm(len(X))]
        return X

    def pdf(self, points):
        sub_pdfs = torch.stack([measure.pdf(points) for measure in self.measures])
        return torch.tensordot(self.lambdas, sub_pdfs, dims=1)



