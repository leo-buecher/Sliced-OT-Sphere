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
        """Projects a vector onto the tangential space
        
        Args:
            X (Tensor): Shape (*Ns, *ms). The vectors in which to consider the tangential spaces.
            dX (Tensor): Shape (*Ns, *ms). The vectors to project to the tangential spaces. (We actually
                project X + dX)
        
        Returns:
            dX_p (Tensor): Shape (*Ns, *ms). The vectors belonging to the tangential spaces.
        """
        pass

    @abstractmethod
    def exponential(self, X, dX):
        """Takes a vector from the tangential space to the manifold

        Args:
            X (Tensor): Shape (*Ns, *ms). The vectors of reference.
            dX (Tensor): Shape (*Ns, *ms). The tangential vectors, as produced by self.projection_tangent_space
        
        Returns:
            X_new (Tensor): Shape (*Ns, *ms). The new vectors of the manifold corresponding to the 
                exponential of each dX in each X.
        """
        pass
    
    @abstractmethod
    def projection_manifold(self, X):
        """Projects a vector on the manifold
        
        Args:
            X (Tensor): Shape (*Ns, *ms). The vectors to project on the manifold.
        
        Returns:
            X_new (Tensor): Shape (*Ns, *ms). The projected vectors.
        """
        pass
    
    @abstractmethod
    def parametrisation(self, A):
        """Returns a vector on the manifold from a vector in R^d
        
        Args:
            A (Tensor): Shape (*As, d) where d is the dimension of the coordinate space. Coordinates of
                parametrised points
        
        Returns:
            X (Tensor): Shape (*As, *ms). Points on the manifold.
        """
        pass
    
    @abstractmethod
    def sample_uniform(self, n):
        """Returns a sample from the uniform probability distribution on the manifold
        
        Args:
            n (int): number of samples
        
        Returns:
            X (Tensor): Shape (n, *ms). Points on the manifold, uniformely distributed.
        """
        pass
    
    @abstractmethod
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
        pass
    
    def sample_projected_gaussian(self, coords, stds, N):
        """Generates one a several samples of gaussians projected on the manifold.

        Args:
            coords (Tensor, np.ndarray, list): Shape (*Ys, *ms). Coordinates of the center(s) of the gaussian(s).
            stds (Tensor, np.ndarray, list, int): Shape (*s) where Ys=(..., *s).
                Standard deviations of the different gaussians.
            N (int): number of samples in each set.

        Returns:
            Tensor: Shape (*Ys, N, *ms)
        """
        if type(coords) != torch.Tensor:
            coords = torch.tensor(coords, dtype=torch.float)
        # M = coords.shape[0]
        Ys = coords.shape[:-self.ndim]
        coords = unsqueeze(coords, len(Ys), 1)

        if type(stds) != torch.Tensor:
            stds = torch.tensor(stds, dtype=torch.float)
        stds = unsqueeze(stds, 0, len(Ys) - stds.ndim) # broadcastable to (*Ys,)
        stds = unsqueeze(stds, -1, 1 + self.ndim) # broadcastable to (*Ys, N, *ms)

        Y = (torch.randn(*Ys, N, *self.ms) * stds) + coords
        Y = self.projection_manifold(Y)
        return Y
    
    @abstractmethod
    def plot_samples(self, Y, X, depthshade=True):
        """Represents samples on the manifold
        
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
        """Projects the samples on the 1D axes corresponding to the different psis

        Args:
            psis (Tensor): Shape (n, *ms) where ms is the shape of the elements of the manifold
            samples (Tensor): Shape (*Ns, *ms) where Ns is a any shape, and prod(Ns) gives the number of samples
        
        Returns:
            projections (Tensor): Shape (*Ns, n)
        """
        pass
    
    @abstractmethod
    def operator_gradient_factor(self, psis, samples):
        """Computes the gradient_factor of operator in every sample, for every psi. The gradient is then
        gardient_factor * psi.

        Args:
            psis (Tensor): Shape (n, *ms) where ms is the shape of the elements of the manifold
            samples (Tensor): Shape (*Ns, *ms) where Ns is any shape, and prod(Ns) gives the number of samples
        
        Returns:
            gradient_factors (Tensor): Shape (*Ns, n) or broadcastable to it
        """
        pass

class ScaManifold(SlicedManifold):
    """Abstract class computing Sliced Wasserstein Barycentres, using the scalar product as operator 
    projecting the measures to the real line and can compute the stochastic gradient descent loop 
    giving the corresponding sliced Wasserstein barycentre.

    It is designed to be used in cooperative inheritance, along with a specific manifold.
    """
    def __init__(self):
        super().__init__()
    
    def operator(self, psis, samples):
        """Projects the samples on the 1D axes corresponding to the different psis

        Args:
            psis (Tensor): Shape (n, *ms) where ms is the shape of the elements of the manifold
            samples (Tensor): Shape (*Ns, *ms) where Ns is a any shape, and prod(Ns) gives the number of samples
        
        Returns:
            projections (Tensor): Shape (*Ns, n)
        """
        return torch.tensordot(samples, psis, dims=((self.dim_range), (self.dim_range))) # simple scalar product of every pair of vector
    
    def operator_gradient_factor(self, psis, samples):
        """Computes the gradient_factor of operator in every sample, for every psi. The gradient is then
        gardient_factor * psi.

        Args:
            psis (Tensor): Shape (n, *ms) where ms is the shape of the elements of the manifold
            samples (Tensor): Shape (*Ns, *ms) where Ns is any shape, and prod(Ns) gives the number of samples
        
        Returns:
            gradient_factors (Tensor): Shape (*Ns, n)
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



