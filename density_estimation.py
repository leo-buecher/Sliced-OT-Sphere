import torch
from src.sphere2 import Sphere, SphereDiscretisation

class DensityEstimationSphere():
    """Class implementing kernel density estimation using the method of [1].
    
    [1] Hall, Peter, G. S. Watson, and Javier Cabrera. "Kernel density estimation with spherical data." Biometrika 74.4 (1987): 751-762."""
    def __init__(self):
        self.kappa = None

    def functional(self, points, kappa0=1000):
        n = points.size(0)
        sca_prod = points @ points.T #(n,n)
        norm_sum = torch.norm(points[:, None] + points[None,:], dim=2) #(n,n)
        mask = 1 - torch.diag(torch.ones((n,)))

        def false_sinh(x):
            return (1 - torch.exp(-2*x))/2
            #return (torch.exp(x) - torch.exp(-x))/2
        
        def f(kappa):
            if type(kappa) != torch.Tensor:
                kappa = torch.Tensor(kappa)

            kappa_squeeze = kappa.clone()
            if kappa.ndim > 0:
                kappa = kappa.unsqueeze(-1).unsqueeze(-1)

            fsk = false_sinh(kappa_squeeze)
            s1 = torch.sum(mask * torch.exp(kappa * (sca_prod - 1)), dim=(-2, -1)) * 2/(1 - 1/n) / fsk
            
            fs2 = (torch.exp(kappa * (norm_sum - 2)) - torch.exp(-kappa * (norm_sum + 2)))/2
            #fs2 = false_sinh(kappa * norm_sum)
            s2 = torch.sum(fs2 / norm_sum, dim=(-2, -1))/ fsk**2
            return 1/(4*torch.pi*n**2) * kappa_squeeze * (s1 - s2)

        #kappas = torch.linspace(kappa0/100, kappa0, 100)[:, None, None]
        #plt.plot(kappas[:, 0, 0], f(kappas))
        return f
            
        
    
    def fit(self, points):
        """Estimates best kappa, for the vMF kernel, on the 2-sphere, in the sense of CV2.
        See Hall, Peter, G. S. Watson, and Javier Cabrera. "Kernel density estimation with spherical data." Biometrika 74.4 (1987): 751-762.
        
        Args:
            points (tensor): (n, 3)
        """
        self.points = points
        f = self.functional(points)

        kappa0 = 1000
        kappas = torch.linspace(kappa0/100, kappa0, 100)
        amax = torch.argmax(f(kappas))
        if amax == 0:
            kappas = kappas/90
            amax = torch.argmax(f(kappas))
        elif amax == len(kappas) - 1:
            kappas = kappas * 90
            amax = torch.argmax(f(kappas))
        else:
            kappas = torch.linspace(kappas[amax - 1], kappas[amax+1], 100)
            amax = torch.argmax(f(kappas))
        self.kappa = float(kappas[amax])

    def predict(self, points, adapted_normalisation=False):
        v = torch.exp(self.kappa * (points @ self.points.T - 1)).mean(1)
        c = self.kappa / (2 * torch.pi) / (1 - torch.exp(-2*torch.tensor(self.kappa)))

        if adapted_normalisation:
            c /= torch.norm(c * v)
        return c * v

    
    def plot(self, N1=100, N2=40, projection="3d", contour=False):
        sphere = Sphere()
        sphere.plot_samples(self.points[None])

        sd = SphereDiscretisation(N1, N2)
        points = sd.build_mesh()
        sd.plot(self.predict(points), projection=projection, contour=contour)
        

        
        


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sphere = Sphere()
    # n_values = torch.logspace(1, 3, 15).long()
    # k_values = [10, 20, 50, 100, 200]
    # k_est = torch.zeros((len(n_values), len(k_values) + 1))
    # 
    # for i, n in enumerate(n_values):
    #     
    #     points = sphere.sample_vMF(torch.tensor([[1., 0, 0]]*len(k_values)), k_values, n)
    #  
    #     points_uni = sphere.sample_uniform(n)
    #     des = DensityEstimationSphere()
    #     des.fit(points_uni)
    #     k_est[i, 0] = des.kappa
    #  
    #     for j, k in enumerate(k_values):
    #         des  = DensityEstimationSphere()
    #         des.fit(points[j])
    #         k_est[i, j+1] = des.kappa
    # 
    # labels = ["uni"] + [f"$\\kappa = {k}$" for k in k_values]
    # for j in range(len(k_values) + 1):
    #     plt.plot(n_values, k_est[:, j], label=labels[j])
    # plt.legend()
    # plt.show()
    #       
    # for j in range(len(k_values) + 1):
    #     plt.loglog(n_values, k_est[:, j], label=labels[j])
    # plt.legend()
    # plt.show()
    
    points = sphere.sample_vMF(torch.tensor([1., -1, 0])/torch.sqrt(torch.tensor(2)), 10, 10)
    #points = sphere.sample_uniform(20)
    des = DensityEstimationSphere()
    des.fit(points)
    des.plot(N1 = 200, N2 = 60, projection="mollweide", contour=True)
