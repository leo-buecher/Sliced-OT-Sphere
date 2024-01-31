import torch
from math import sqrt, pi

import matplotlib.pyplot as plt
import time

from src.sphere2 import Sphere, ScaSphere, UniformSphere, UniformPortionSphere, VMFSphere, SmileySphere, SphereDiscretisation
from src.barycenters import LSWBarycenter, ESWBarycenter, WBarycenter, WRegBarycenter
from src.ssb import SSWBarycenter

from density_estimation import DensityEstimationSphere
from test import Test, logspace2_man
from utils import tqdm, USE_TQDM, repr2
from platform import node as platform_node


import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)


def expe(Y, n_psis=1000, max_iter=1000, X0=None, lambdas=None, tau=0.5):
    """Computes the LPSBarycenter of the given measures on the sphere, plot it and displays the loss evolution"""
    sca_sphere = ScaSphere()
    swb = LSWBarycenter(sca_sphere)
    if X0 is None:
        X0 = sca_sphere.sample_uniform(Y.shape[1])
    
    sca_sphere.plot_samples(Y, X0)

    swb.fit(Y, lambdas, X0, n_psis, tau, max_iter, compute_objective=True)
    swb.plot_samples()

    plt.plot(swb.L_loss)
    plt.show()

def dirac_problem(X0, n_psis=1000, max_iter=1000, tau=0.5):
    """Computes the LPSBarycenter of two antipodal dirac measures on the sphere, plots it and the loss evolution"""
    Y = torch.zeros((2, X0.shape[0], 3))
    Y[0] = torch.tensor([0, 0, 1] , dtype=torch.float) # broadcasting
    Y[1] = torch.tensor([0, 0, -1], dtype=torch.float) 

    expe(Y, n_psis=n_psis, max_iter=max_iter, X0=X0, tau=tau)

def test_sliced_barycenter2(N=50, n_psis=500, max_iter=1000, tau=2):
    """Computes the LPSBarycenter of two vMF distant by an angle of pi/4 on the sphere, plots it, the loss evolution
    compared with the expected loss, and the step norm evolution."""
    # N: int or tuple (Nx, list [Nx_j for j])
    gnrl = type(N)== tuple
    if gnrl:
        Nx = N[0]
        Nys = N[1]
    else:
        Nx = N
    
    sca_sphere = ScaSphere()
    swb = LSWBarycenter(sca_sphere)
    coords = torch.tensor([[1, 1, 0], [1, -1, 0]], dtype=torch.float)/sqrt(2)
    kappas = torch.tensor([100, 100], dtype=torch.float)
    if gnrl:
        Y = [sca_sphere.sample_vMF(coords[j], kappas[j], Nys[j]) for j in range(2)]
    else:
        Y = sca_sphere.sample_vMF(coords, kappas, N)

    X0 = sca_sphere.sample_uniform(Nx)
    sca_sphere.plot_samples(Y, X0)

    swb.fit(Y, None, X0, n_psis=n_psis, tau_init=tau, max_iter=max_iter)
    swb.plot_samples()

    plt.plot(swb.L_loss)
    for i in range(50):
        plt.axhline(1/4 * swb.sliced_dist_square(sca_sphere.sample_uniform(n_psis), Y[0], Y[1]), c="r", lw=0.5)

    # plt.plot(3*np.array(swb.L_step)+0.2)
    # a = np.array(swb.L_stop)*100
    # plt.plot(np.arange(len(a)) + 40, a[:, 0])
    # plt.plot(np.arange(len(a)) + 40, a[:, 1])
    plt.grid()
    plt.show()

    plt.plot(swb.L_step)
    plt.grid()
    plt.show()


###################################################################################

class TauExpe(Test):
    """Study the influence of tau (constant) on the convergence"""
    def __init__(self, method="paral", tau_min=0.01, tau_max=100, N=40, n_psis=500, max_iter=1000, stop=True):
        super().__init__()
        self.method   = method  
        self.tau_min  = tau_min 
        self.tau_max  = tau_max 
        self.N        = N       
        self.n_psis   = n_psis  
        self.max_iter = max_iter
        self.stop     = stop    

        self.tau_values = logspace2_man(self.tau_min, self.tau_max*2)
        self.Ls_loss = []
        self.Ls_diff = []

    @property
    def name(self):
        return f"expe_tau_{self.method}_tau{self.tau_min}-{self.tau_max}_N{self.N}_n{self.n_psis}_mi{self.max_iter}"
    
    def run(self):
        self.node = platform_node()
        sphere = Sphere()
        coords = torch.tensor([[1, 1, 0], [1, -1, 0]], dtype=torch.float)/sqrt(2)
        kappas = torch.tensor([100, 100], dtype=torch.float)
        Y = sphere.sample_vMF(coords, kappas, self.N)

        X0 = sphere.sample_uniform(self.N)
        for tau in tqdm(self.tau_values):
            swb = LSWBarycenter(ScaSphere()) if self.method=="paral" else SSWBarycenter()
            swb.fit(Y, None, X0, n_psis=self.n_psis, tau_init=tau, max_iter=self.max_iter, tqdm_leave=False, stop=self.stop)
            self.Ls_loss.append(swb.L_loss)
            self.Ls_diff.append(swb.L_step)
        self.completed = True
    
    def plot(self):
        print("Loss")
        for i, tau in enumerate(self.tau_values):
            # if i%3: continue
            plt.plot(self.Ls_loss[i], label=f"$\\tau={tau:.1}$")
            # print(f"{tau:6.5} {np.mean(self.Ls_loss[i][-100:]):.6f} {np.std(self.Ls_loss[i][-100:]):.6f}")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()
        # from scipy.stats import linregress
        # for i, tau in enumerate(self.tau_values):
        #     if i%3: continue
        #     slopes=[]
        #     #fixed for l in range(2, 20):
        #     #fixed     reg = linregress(np.arange(l), self.Ls_loss[i][:l], alternative="less")
        #     #fixed     slopes.append([reg.slope, reg.stderr])
        #     #fixed for l in range(20, 40):
        #     #fixed     reg = linregress(np.arange(20), self.Ls_loss[i][l-20:l], alternative="less")
        #     #fixed     slopes.append([reg.slope, reg.stderr])
        #     #fixed for l in range(40, len(self.Ls_loss[i])):
        #     slopes_deb = []
        #     for l in range(3, len(self.Ls_loss[i])):
        #         reg = linregress(np.arange(l-(l//2)), self.Ls_loss[i][(l//2):l], alternative="less")
        #         slopes.append([reg.slope, reg.stderr])
        #         reg = linregress(np.arange(l-(l//2)), self.Ls_loss[i][:l-(l//2)], alternative="less")
        #         slopes_deb.append([reg.slope, reg.stderr])
        #     slopes = np.array(slopes)
        #     slopes_deb = np.array(slopes_deb)
        #     l, = plt.plot(slopes[:, 0], label=f"$\\tau={tau}$")
        #     plt.fill_between(np.arange(len(slopes)), slopes[:, 0] + slopes[:,1], slopes[:, 0] - slopes[:, 1], alpha=0.5)
        #     plt.plot(slopes_deb[:, 0], ":", c=l.get_c())
        #     plt.fill_between(np.arange(len(slopes_deb)), slopes_deb[:, 0] + slopes_deb[:,1], slopes_deb[:,0]-slopes_deb[:,1], alpha=0.5, color=l.get_c())
        #     #fixed plt.plot([slopes[18, 0]/20]*len(slopes), color=l.get_c())
        #     #fixed plt.fill_between(np.arange(len(slopes)), [slopes[18, 0]/20]*len(slopes), [slopes[18, 0]/20 + slopes[18, 1]/20]*len(slopes), color=l.get_c(), alpha=0.5)
        # plt.axvline(17, c="k")
        # #fixed plt.axvline(18, c="k")
        # #fixed plt.axvline(38, c="k")
        # plt.grid()
        # plt.legend()
        # plt.show()
        print("\nStep norm")
        for i, tau in enumerate(self.tau_values):
            plt.plot(self.Ls_diff[i], label=f"$\\tau={tau:.1}$")
            # print(f"{tau:6.5} {np.mean(self.Ls_diff[i][-100:]):.6f} {np.std(self.Ls_diff[i][-100:]):.6f}")
        plt.xlabel("Iteration")
        plt.ylabel("Step norm")
        plt.legend()
        plt.grid()
        plt.show()

class TauDecLagranExpe(Test):
    """Study the influence of tau on the convergence of the LPSB algorithm. It follows the law
    $$\\tau_l = \\frac{\\tau_0}{(al + 1)^p}$$
    and the influence of the three parameters $\\tau_0$, $a$ and $p$ is exhibited."""
    def __init__(self, method="paral", N=40, n_psis=500, max_iter=1000):
        super().__init__()
        self.method   = method
        self.N        = N       
        self.n_psis   = n_psis  
        self.max_iter = max_iter

        self.tau_values = [1., 20., 100., 100., 100., 100.]
        self.a_values   = [1.,  1.,   1.,   1.,   1.,  0.1]
        self.pow_values = [0.,  0.,   0.,   1.,   2.,   1.]
        self.Ls_loss = []
        self.Ls_diff = []
    
    @property
    def name(self):
        return f"expe_taudecl_{self.method}_N{self.N}_n{self.n_psis}_mi{self.max_iter}"
    
    def run(self):
        self.node = platform_node()
        sphere = Sphere()
        coords = torch.tensor([[1, 1, 0], [1, -1, 0]], dtype=torch.float)/sqrt(2)
        kappas = torch.tensor([100, 100], dtype=torch.float)
        Y = sphere.sample_vMF(coords, kappas, self.N)

        X0 = sphere.sample_uniform(self.N)
        for tau_init, a_tau, pow_tau in zip(tqdm(self.tau_values), self.a_values, self.pow_values):
            swb = LSWBarycenter(ScaSphere()) if self.method=="paral" else SSWBarycenter()
            swb.fit(Y, None, X0, n_psis=self.n_psis, tau_init=tau_init, max_iter=self.max_iter, a_tau=a_tau, pow_tau=pow_tau, tqdm_leave=False, stop=False)
            self.Ls_loss.append(swb.L_loss)
            self.Ls_diff.append(swb.L_step)
        self.completed = True
    
    def plot(self):
        for i, (tau, a_tau, pow_tau) in enumerate(zip(self.tau_values, self.a_values, self.pow_values)):
            plt.plot(self.Ls_loss[i], label=f"$\\tau={tau:.1}, a={a_tau:.1}, p={pow_tau:.1}$")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

        for i, (tau, a_tau, pow_tau) in enumerate(zip(self.tau_values, self.a_values, self.pow_values)):
            plt.plot(self.Ls_diff[i], label=f"$\\tau={tau:.1}, a={a_tau:.1}, p={pow_tau:.1}$")
        plt.xlabel("Iteration")
        plt.ylabel("Step norm")
        plt.legend()
        plt.grid()
        plt.show()

class TauDecEulerExpe(Test):
    """Study the influence of tau on the convergence of the EPSB algorithm. It follows the law
    $$\\tau_l = \\frac{\\tau_0}{(al + 1)^p}$$
    and the influence of the three parameters $\\tau_0$, $a$ and $p$ is exhibited."""
    def __init__(self, N1=80, N2=25, n_psis=500, max_iter=30):
        super().__init__()
        self.N1 = N1
        self.N2 = N2
        self.n_psis = n_psis
        self.max_iter = max_iter

        self.pow_values = [0, 1, 2]
        self.a_values = [0.05, 0.2]
        self.tau_init_values = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3]

        shape = (len(self.pow_values), len(self.a_values), len(self.tau_init_values), max_iter)
        self.L_losses = torch.zeros(shape)
        self.L_step_norms = torch.zeros(shape)

    @property
    def name(self):
        return f"expe_taudece_N{self.N1}-{self.N2}_n{self.n_psis}_mi{self.max_iter}"
    
    def run(self):
        self.node = platform_node()
        sd = SphereDiscretisation(self.N1, self.N2)
        X = sd.build_mesh()
        N = len(X)
        coords = torch.tensor([[1, -1, 0], [1, 1, 0]])/sqrt(2)
        V = torch.exp(10 * coords @ X.T)

        V /= torch.sum(V, axis=1, keepdim=True)
        W0 = torch.ones(N)/N
        for i, pow in enumerate(tqdm(self.pow_values, leave=True)):
            for j, a in enumerate(tqdm(self.a_values, leave=False)):
                for k, tau_init in enumerate(tqdm(self.tau_init_values, leave=False)):
                    if pow > 0 or j == 0:
                        eswb = ESWBarycenter(ScaSphere())
                        eswb.fit(X, V, None, W0, n_psis=self.n_psis, tau_init=tau_init, max_iter=self.max_iter, a_tau=a, pow_tau=pow, stop=False, tqdm_leave=False)
                        self.L_losses[i, j, k, :] = torch.Tensor(eswb.L_loss)
                        self.L_step_norms[i, j, k, :] = torch.Tensor(eswb.L_step)
                    else:
                        self.L_losses[i, j, k, :] = torch.nan
                        self.L_step_norms[i, j, k, :] = torch.nan
        self.completed = True
    
    def plot(self):
        markers = ["", "o", "+"]
        lines= ["-", ":"]
        for i, pow in enumerate(self.pow_values):
            for j, a in enumerate(self.a_values):
                for k, tau_init in enumerate(self.tau_init_values):
                    if tau_init == 1e-3:
                        display_label = (i== 0 and j==0) or (j==0 and k==0) or (k==0 and i==0)
                        label = f"$\\tau={tau_init}, a={a}, p={pow}$"
                        plt.plot(self.L_losses[i, j, k], c=f"C{k}", marker=markers[i], ls=lines[j], label=label if display_label else None)
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, "both")
        plt.show()

        for i, pow in enumerate(self.pow_values):
            for j, a in enumerate(self.a_values):
                for k, tau_init in enumerate(self.tau_init_values):
                    if tau_init == 1e-3:
                        display_label = (i== 0 and j==0) or (j==0 and k==0) or (k==0 and i==0)
                        label = f"$\\tau={tau_init}, a={a}, p={pow}$"
                        plt.plot(self.L_step_norms[i, j, k], c=f"C{k}", marker=markers[i], ls=lines[j], label=label if display_label else None)
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Step norm")
        plt.grid(True, "both")
        plt.show()


class ConvergenceExpe(Test):
    """Compare the convergence of LPSB and SSB algorithms for the case of 2vMF 
    - with centers Phi(pi/4, pi/2), Phi(-pi/4, pi/2)
    - and kappa=100
    """
    def __init__(self, N=50, n_psis=500, max_iter=1000, tau=[20, 50]):
        super().__init__()
        self.N = N
        # N: int or tuple (Nx, list [Nx_j for j])
        self.n_psis=n_psis
        self.max_iter=max_iter
        if type(tau) != list:
            self.tau = [tau, tau]
        else:
            self.tau = tau

        self.measures = None
        self.X0 = None
        self.pbarycenter = None
        self.pL_loss = None
        self.pL_step = None
        self.sbarycenter = None
        self.sL_loss = None
        self.sL_step = None
        
    
    @property
    def name(self):
        return f"expe_cvrg_N{self.N}_n{self.n_psis}_mi{self.max_iter}_tau{self.tau[0]}-{self.tau[1]}"

    def run(self):
        self.node = platform_node()
        sphere = Sphere()

        # rewriting input parameters
        gnrl = type(self.N)== tuple #whether to used the generalised algorithm version
        if gnrl:
            Nx = self.N[0]
            Nys = self.N[1]
        else:
            Nx = self.N
        
        # defining input measures
        coords = torch.tensor([[1, 1, 0], [1, -1, 0]], dtype=torch.float)/sqrt(2)
        kappas = torch.tensor([100, 100], dtype=torch.float)
        if gnrl:
            self.measures = [sphere.sample_vMF(coords[j], kappas[j], Nys[j]) for j in range(2)]
        else:
            self.measures = sphere.sample_vMF(coords, kappas, self.N)

        # defining initialisation
        self.X0 = sphere.sample_uniform(Nx)

        # running experiments
        psb = LSWBarycenter(ScaSphere())
        ssb = SSWBarycenter()
        psb.fit( self.measures, None, self.X0, n_psis=self.n_psis, tau_init=self.tau[0], max_iter=self.max_iter)
        ssb.fit(self.measures, None, self.X0, n_psis=self.n_psis, tau_init=self.tau[1], max_iter=self.max_iter)

        # saving results
        self.pbarycenter = psb.barycenter
        self.pL_loss = psb.L_loss
        self.pL_step = psb.L_step
        self.sbarycenter = ssb.barycenter
        self.sL_loss = ssb.L_loss
        self.sL_step = ssb.L_step

        self.completed = True
    
    def plot(self):
        # print parameters
        for k in ["node", "N", "n_psis", "max_iter", "tau"]:
            print(f"{k:8}: {repr2(self.__getattribute__(k))}")
        
        # plot results
        sphere = Sphere()
        # sphere.plot_samples(self.measures, self.X0)
        sphere.plot_samples(self.measures, self.pbarycenter)
        sphere.plot_samples(self.measures, self.sbarycenter)

        plt.figure(self.name + "_loss")
        plt.plot(self.pL_loss, label="Parallel")
        plt.plot(8.5*torch.tensor(self.sL_loss), label="Semi-circular")
        # for i in range(50):
        #     plt.axhline(1/4 * swb.sliced_dist_square(swb.sample_uniform(n_psis), Y[0], Y[1]), c="r", lw=0.5)
        plt.legend()
        # plt.title("(Rescaled) Loss evolution over the iterations")
        plt.grid()
        plt.show()

        plt.figure(self.name + "_step")
        plt.plot(self.pL_step, label="Parallel")
        plt.plot(self.sL_step, label="Semi-circular")
        plt.legend()
        # plt.title("Step norm for each iteration")
        plt.grid()
        plt.show()


class TimeExpe(Test):
    """Compare the execution times of the LPSB and SSB algorithms with the same number of iterations, for the case of 2 vMF distributions,
    in terms of influence of the number N of points and the number n_psi of slices."""
    def __init__(self, N_def=40, N_max=100, n_psis_def=200, n_psis_max=500, max_iter=100):
        super().__init__()
        self.N_def = N_def
        self.N_max = N_max
        self.n_psis_def = n_psis_def
        self.n_psis_max = n_psis_max
        self.max_iter = max_iter
        self.tau = 20

        self.N_values = logspace2_man(N_max*2)
        self.n_psis_values = logspace2_man(n_psis_max*2)
        self.times_N = torch.zeros((len(self.N_values), 3))
        self.times_n_psis = torch.zeros((len(self.n_psis_values), 3))
    
    @property
    def name(self):
        return f"expe_time_N{self.N_def}-{self.N_max}_n{self.n_psis_def}-{self.n_psis_max}_mi{self.max_iter}"
    
    def run(self):
        self.node = platform_node()
        sphere = Sphere()
        ssb = SSWBarycenter()
        psb = LSWBarycenter(ScaSphere())

        coords = torch.tensor([[1, 1, 0], [1, -1, 0]], dtype=torch.float)/sqrt(2)
        kappas = torch.tensor([100, 100], dtype=torch.float)

        for i, N in enumerate(tqdm(self.N_values)):
            X0 = sphere.sample_uniform(N)
            Y = sphere.sample_vMF(coords, kappas, N)

            t0 = time.time()
            psb.fit(Y, None, X0, n_psis=self.n_psis_def, tau_init=self.tau, max_iter=self.max_iter, tqdm_leave=False)
            t1 = time.time()
            self.times_N[i, 0] = t1 - t0

            Y = [Y[j] for j in range(2)]
            t0 = time.time()
            psb.fit(Y, None, X0, n_psis=self.n_psis_def, tau_init=self.tau, max_iter=self.max_iter, tqdm_leave=False)
            t1 = time.time()
            self.times_N[i, 1] = t1 - t0

            t0 = time.time()
            ssb.fit(Y, None, X0, n_psis=self.n_psis_def, tau_init=self.tau, max_iter=self.max_iter, tqdm_leave=False)
            t1 = time.time()
            self.times_N[i, 2] = t1 - t0
        
        X0 = sphere.sample_uniform(self.N_def)
        Y = sphere.sample_vMF(coords, kappas, self.N_def)
        for i, n_psis in enumerate(tqdm(self.n_psis_values)):

            t0 = time.time()
            psb.fit(Y, None, X0, n_psis=n_psis, tau_init=self.tau, max_iter=self.max_iter, tqdm_leave=False)
            t1 = time.time()
            self.times_n_psis[i, 0] = t1 - t0

            Y = [Y[j] for j in range(2)]
            t0 = time.time()
            psb.fit(Y, None, X0, n_psis=n_psis, tau_init=self.tau, max_iter=self.max_iter, tqdm_leave=False)
            t1 = time.time()
            self.times_n_psis[i, 1] = t1 - t0

            t0 = time.time()
            ssb.fit(Y, None, X0, n_psis=n_psis, tau_init=self.tau, max_iter=self.max_iter, tqdm_leave=False)
            t1 = time.time()
            self.times_n_psis[i, 2] = t1 - t0
            Y = torch.stack(Y)
        self.completed = True
    
    def plot(self):
        # print parameters
        for k in ["node", "N_def", "N_max", "n_psis_def", "n_psis_max", "max_iter", "tau"]:
            print(f"{k:10}: {repr(self.__getattribute__(k))}")
        
        # plot results
        plt.figure(f"{self.name}_wrt_N")
        plt.loglog(self.N_values, self.times_N[:, 0], label="Parallel")
        # plt.loglog(self.N_values, self.times_N[:, 1], label="Generalised parallel")
        plt.loglog(self.N_values, self.times_N[:, 2], label="Semi-circular")
        # plt.loglog(self.N_values, 0.03*self.N_values)
        plt.legend()
        plt.grid(True, which="both")
        plt.xlabel("N")
        plt.ylabel("Time (s)")
        plt.show()

        plt.figure(f"{self.name}_wrt_npsi")
        plt.loglog(self.n_psis_values, self.times_n_psis[:, 0], label="Parallel")
        # plt.loglog(self.n_psis_values, self.times_n_psis[:, 1], label="Generalised parallel")
        plt.loglog(self.n_psis_values, self.times_n_psis[:, 2], label="Semi-circular")
        plt.legend()
        plt.grid(True, which="both")
        plt.xlabel("P")
        plt.ylabel("Time (s)")
        plt.show()


class ShapeExpe(Test):
    """Compare the shapes of the LPSBarycenter and SSBarycenter for different input settings."""
    def __init__(self, N=200, n_psis=500, max_iter=1000, tau=[40, 80]):
        super().__init__()
        self.N = N
        # N: int or tuple (Nx, list [Nx_j for j])
        self.n_psis=n_psis
        self.max_iter=max_iter
        if type(tau) == list:
            self.tau = tau
        else:
            self.tau = [tau, tau]
        
        self.input_params = {
            "2vMF"     : [["vMF", [1, 1, 0], 100], ["vMF", [1, -1, 0], 100]],
            "2antipod" : [["vMF", [0, 0, 1], 400], ["vMF", [0, 0, -1], 400]],
            "RingVMF"  : [["vMF", [0, 0, 1], 400], ["uniform", [[0, 2*pi], [-0.1, 0.1]]]],
            "RingCrois": [["uniform", [[              0,            2*pi], [-0.1, 0.1]]], ["uniform", [[-pi/10, pi/10], [-1, 1]]]],
            "2crois180": [["uniform", [[        9*pi/10,        11*pi/10], [  -1,   1]]], ["uniform", [[-pi/10, pi/10], [-1, 1]]]],
            "2crois120": [["uniform", [[(4/3 - 1/10)*pi, (4/3 + 1/10)*pi], [  -1,   1]]], ["uniform", [[-pi/10, pi/10], [-1, 1]]]],
        }

        self.Y_list = {}
        # self.X_paral_list = []
        # self.X_scirc_list = []
        self.X_list = {"PSB":{}, "SSB":{}}
        self.times  = {"PSB":{}, "SSB":{}}
        self.L_loss = {"PSB":{}, "SSB":{}}
        self.X0 = None
    
    @property
    def name(self):
        return f"expe_shape_N{self.N}_n{self.n_psis}_mi{self.max_iter}_tau{self.tau[0]}-{self.tau[1]}"

    def run(self):
        self.node = platform_node()
        gnrl = type(self.N)== tuple
        if gnrl:
            Nx = self.N[0]
            Nys = self.N[1]
        else:
            Nx = self.N
            Nys = [self.N, self.N]

        param_to_measure = lambda param: (VMFSphere if param[0] == "vMF" else UniformPortionSphere)(*param[1:])
        
        sphere = Sphere()
        self.X0 = sphere.sample_uniform(Nx)
        # for m1, m2 in measures_pairs:
        for name, (input1, input2) in self.input_params.items():
            m1 = param_to_measure(input1)
            m2 = param_to_measure(input2)
            Y = [m1.sample(Nys[0]), m2.sample(Nys[1])]
            if not gnrl:
                Y = torch.stack(Y)
            self.Y_list[name] = Y

            psb = LSWBarycenter(ScaSphere())
            ssb = SSWBarycenter()

            t0 = time.time()
            psb.fit(Y, None, self.X0, n_psis=self.n_psis, tau_init=self.tau[0], max_iter=self.max_iter)
            self.times["PSB"][name] = time.time() - t0
            self.L_loss["PSB"][name] = psb.L_loss
            self.X_list["PSB"][name] = psb.barycenter

            t0 = time.time()
            ssb.fit(Y, None, self.X0, n_psis=self.n_psis, tau_init=self.tau[1], max_iter=self.max_iter)
            self.times["SSB"][name] = time.time() - t0
            self.L_loss["SSB"][name] = ssb.L_loss
            self.X_list["SSB"][name] = ssb.barycenter

        self.completed = True
    
    def plot(self):
        sphere = Sphere()

        #print parameters and time results
        for k in ["node", "N", "n_psis", "max_iter", "tau", "times"]:
            print(f"{k:8}: {repr2(self.__getattribute__(k))}")
        
        #plot results
        for input_name in self.input_params.keys():
            sphere.plot_samples(self.Y_list[input_name], self.X0, figname=f"{self.name}_{input_name}_X0")

            sphere.plot_samples(self.Y_list[input_name], self.X_list["PSB"][input_name], figname=f"{self.name}_{input_name}_PSB")
            if len(self.L_loss["PSB"]) == len(self.input_params):
                plt.figure(f"{self.name}_{input_name}_PSB_loss")
                plt.plot(self.L_loss["PSB"][input_name])
                plt.show()

            sphere.plot_samples(self.Y_list[input_name], self.X_list["SSB"][input_name], figname=f"{self.name}_{input_name}_SSB")
            if len(self.L_loss["SSB"]) == len(self.input_params):
                plt.figure(f"{self.name}_{input_name}_SSB_loss")
                plt.plot(self.L_loss["SSB"][input_name])
                plt.show()


class EPSBExpe(Test):
    def __init__(self, inputs="SmileyVMF", N1=200, N2=60, n_psis=200, tau=.001, max_iter=100, a_tau=0.1, pow_tau=0):
        """Shows the results of the EPSB algorithm in terms of barycenter, loss and step evolution.

        Args:
            inputs (str, optional): Input setting: 2vMF, 2antipod, 2antipodv2, RingVMF, SmileyVMF. Defaults to "SmileyVMF".
            ...
        """
        super().__init__()
        assert inputs in ["2vMF", "2antipod", "2antipodv2", "RingVMF", "SmileyVMF"]
        self.inputs         = inputs        
        self.N1             = N1            
        self.N2             = N2            
        self.n_psis         = n_psis        
        self.tau            = tau           
        self.max_iter       = max_iter      
        # self.representation = representation
        self.a_tau          = a_tau         
        self.pow_tau        = pow_tau       

        self.sd_dict = None
        self.V = None
        self.W0 = None
        self.W =  None #barycenter
        self.L_loss = None
        self.L_step = None
    
    @property
    def name(self):
        return f"expe_EPSB_{self.inputs}_N{self.N1}-{self.N2}_n{self.n_psis}_tau{self.tau}_mi{self.max_iter}_a{self.a_tau}_p{self.pow_tau}"
        
    def run(self):
        self.node = platform_node()
        eswb = ESWBarycenter(ScaSphere())

        # torch.manual_seed(0)
        # X = sphere.sample_uniform(N)
        sd = SphereDiscretisation(self.N1, self.N2)
        X = sd.build_mesh()
        self.sd_dict = sd.__dict__
        N = len(X)

        if self.inputs == "2vmf":
            coords = torch.tensor([[1, -1, 0], [1, 1, 0]])/sqrt(2)
            self.V = torch.exp(10 * coords @ X.T)
        elif self.inputs == "2antipod":
            coords = torch.tensor([[0, 0, 1], [0, 0, -1.]])
            self.V = torch.exp(40 * coords @ X.T)
        elif self.inputs == "2antipodv2":
            self.V = torch.zeros((2, N))
            q = 6
            ma = X[:,2] >= X[torch.argsort(X[:,2])[-q], 2]
            mi = X[:,2] <= X[torch.argsort(X[:,2])[q-1], 2] 
            self.V[0, ma] = 1/q
            self.V[1, mi] = 1/q
        elif self.inputs == "RingVMF":
            coords = torch.tensor([[0, 0, 1.]])
            V0 = torch.exp(10 * coords @ X.T)
            V1 = torch.unsqueeze(torch.exp(-5 * X[:,2]**2), 0)
            self.V = torch.concatenate([V0, V1])
        elif self.inputs == "SmileyVMF":
            V0 = SmileySphere(n_modes_mouth=17).discretise_on(X)[None]
            V1 = VMFSphere([1., 0, 0], 30).discretise_on(X)[None]
            self.V = torch.concatenate([V0, V1])

        self.V /= torch.sum(self.V, axis=1, keepdim=True)
        self.W0 = torch.ones(N)/N
        
        eswb.fit(X, self.V, None, self.W0, n_psis=self.n_psis, tau_init=self.tau, max_iter=self.max_iter, a_tau=self.a_tau, pow_tau=self.pow_tau, stop=False)
        self.W = eswb.barycenter
        self.L_loss = eswb.L_loss
        self.L_step = eswb.L_step
        self.completed = True
    
    def plot(self, representation="3d"):
        assert representation in ["colours_mollweide", "colours_3d", "wire", "3d", "mollweide"]

        sd = SphereDiscretisation(self.N1, self.N2)
        sd.__dict__ = self.sd_dict
        
        if representation.startswith("colours_"):
            sd.plot_colours(self.V, self.W0, projection=representation[8:])
        elif representation == "wire":
            sd.plot_wire(self.V, self.W0)
        elif representation in ["3d", "mollweide"]:
            for i, Vi in enumerate(self.V):
                sd.plot(Vi, projection=representation, figname=f"input {i}")


        if representation.startswith("colours_"):
            sd.plot_colours(self.V, self.W, projection=representation[8:])
        elif representation == "wire":
            sd.plot_wire(self.V, self.W)
        elif representation in ["3d", "mollweide"]:
            sd.plot(self.W, projection=representation, figname="barycentre")

        plt.plot(self.L_loss)
        plt.show()

        plt.plot(self.L_step)
        plt.show()

class BenchmarkExpe(Test):
    """Compare the shapes of the LPSBarycenter, LSSBarycenter (=SSBarycenter), EPSBarycenter, and, if possible the true and the regularised \
    Wasserstein barycenters. 2 input settings are possible: 2vMF or Smiley+vMF"""

    def __init__(self, inputs="SmileyVMF", N=200, N1=150, N2=50, n_psis=500, reg=1e-3):
        """inputs in ['2vMF', 'SmileyVMF']"""
        super().__init__()
        if inputs not in ["2vMF", "SmileyVMF"]:
            raise ValueError(f"{self.inputs} is not a valid name of input settings")
        # Input parameters
        self.inputs = inputs
        self.N = N
        self.N1 = N1
        self.N2 = N2
        self.n_psis = n_psis
        self.reg = reg

        self.param_lpsb = {
            "n_psis"  : self.n_psis,
            "tau_init": 40,
            "max_iter": 1000,
        }
        self.param_lssb = {
            "n_psis"  : self.n_psis,
            "tau_init": 80,
            "max_iter": 1000,
        }
        self.param_epsb = {
            "n_psis"  : self.n_psis,
            "tau_init": 5e-3,
            "a_tau"   : 5e-2,
            "pow_tau" : 0.5,
            "max_iter": 500,
            "stop"    : False,
        }

        self.param_input_smiley = {
            "smiley": {"cpt": [torch.pi/4, 0.8*torch.pi/2], "n_modes_mouth":17},
            "vmf": {"coords": [1, 1, 0.5], "kappa":30},
        }

        self.param_input_2vmf = {
            0: {"coords": [1, 1, 0], "kappa": 30},
            1: {"coords": [1, -1, 0], "kappa": 30},
        }

        # Results
        self.times = {}
        self.Y = None
        self.V = None
        self.lpsbar = None
        self.lssbar = None
        self.epsbar = None
        self.wbar = None
        self.wrbar = None
    
    @property
    def name(self):
        return f"expe_benchmark_{self.inputs}_Nl{self.N}_Ne{self.N1}-{self.N2}_n{self.n_psis}_reg{self.reg}"

    def run(self):
        self.node = platform_node()
        meas0 = UniformSphere()
        if self.inputs == "2vMF":
            meas1 = VMFSphere(**self.param_input_2vmf[0])
            meas2 = VMFSphere(**self.param_input_2vmf[1])
        elif self.inputs == "SmileyVMF":
            meas1 = SmileySphere(**self.param_input_smiley["smiley"])
            meas2 = VMFSphere(**self.param_input_smiley["vmf"])
        else:
            raise ValueError(f"{self.inputs} is not a valid name of input settings")

        sca_sphere = ScaSphere()
        lpsb = LSWBarycenter(sca_sphere)
        lssb = SSWBarycenter()
        epsb = ESWBarycenter(sca_sphere)
        wb = WBarycenter()
        wrb = WRegBarycenter()

        self.Y = torch.stack([meas1.sample(self.N), meas2.sample(self.N)])
        X0 = meas0.sample(self.N)

        sd = SphereDiscretisation(N1=self.N1, N2=self.N2)
        Xupport = sd.build_mesh()
        self.V = torch.stack([meas1.discretise_on(Xupport), meas2.discretise_on(Xupport)])
        W0 = meas0.discretise_on(Xupport)

        t0 = time.time()
        lpsb.fit(self.Y, None, X0, **self.param_lpsb)
        t1 = time.time()
        self.lpsbar = lpsb.barycenter
        self.times["lpsb"] = t1 - t0

        t0 = time.time()
        lssb.fit(self.Y, None, X0, **self.param_lssb)
        t1 = time.time()
        self.lssbar = lssb.barycenter.detach()
        self.times["lssb"] = t1 - t0

        t0 = time.time()
        epsb.fit(Xupport, self.V, None, W0, **self.param_epsb)
        t1 = time.time()
        self.epsbar = epsb.barycenter
        self.times["epsb"] = t1 - t0

        # print("Wasserstein barycentre: ", end="")
        # t0 = time.time()
        # wb.fit(Xupport, self.V, None)
        # t1 = time.time()
        # self.wbar = wb.barycenter
        # self.times["wbar"] = t1 - t0
        # print(t1 -t0, "s")
        
        print("Computing the regularised Wasserstein barycentre")
        t0 = time.time()
        wrb.fit(Xupport, self.V, None, self.reg, verbose=True)
        t1 = time.time()
        self.wrbar = wrb.barycenter
        self.times["wrb"] = t1 - t0
        print("Completed in", t1 - t0, "s")

        self.completed = True

    def plot(self, projection="mollweide", contour=False): # 3d or mollweide
        # print parameters and time results
        for k in ["node", "inputs", "N", "N1", "N2", "reg", "param_lpsb", "param_lssb", "param_epsb", "times"]:
            print(f"{k:10}: {repr2(self.__getattribute__(k))}")
        
        des = DensityEstimationSphere()
        sd = SphereDiscretisation(N1=self.N1, N2=self.N2)
        Xupport = sd.build_mesh()

        # plot results
        des.fit(self.Y[0])
        sd.plot(des.predict(Xupport, adapted_normalisation=True), projection=projection, figname=f"{self.name}_input0l", contour=contour)
        des.fit(self.Y[1])
        sd.plot(des.predict(Xupport, adapted_normalisation=True), projection=projection, figname=f"{self.name}_input1l", contour=contour)
        sd.plot(self.V[0], projection=projection, figname=f"{self.name}_input0e", contour=contour)
        sd.plot(self.V[1], projection=projection, figname=f"{self.name}_input1e", contour=contour)
        
        des.fit(self.lpsbar)
        sd.plot(des.predict(Xupport, adapted_normalisation=True), projection=projection, figname=f"{self.name}_lpsb", contour=contour)
        
        des.fit(self.lssbar)
        sd.plot(des.predict(Xupport, adapted_normalisation=True), projection=projection, figname=f"{self.name}_lssb", contour=contour)
        
        sd.plot(self.epsbar, projection=projection, figname=f"{self.name}_epsb", contour=contour)

        # # sd.plot(self.wbar, projection=projection, figname=f"{self.name}_wb", contour=contour)

        sd.plot(self.wrbar, projection=projection, figname=f"{self.name}_reg_wb", contour=contour)


#####################################################################################

if __name__ == "__main__":
    sca_sphere = ScaSphere()

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-t", "--test-name", type=str, default="visual",
                        help="Name of test to launch. Can be 'visual', 'tau', 'tau_decl', 'tau_dece', 'convergence', 'time', 'shape', 'epsb', 'benchmark'. Defaults to 'visual'.")
    parser.add_argument("-w", "--without-tqdm", action="store_true",
                        help="Prevent tqdm bars")
    args = parser.parse_args()
    test_name = args.test_name
    USE_TQDM[0] = not args.without_tqdm

    if test_name == "visual":
        # test_sliced_barycenter2(N=(40, [40, 40]), n_psis=500, max_iter=1000)
        test_sliced_barycenter2(N=40, n_psis=500, max_iter=1000)

    elif test_name == "tau":
        expe_tau = TauExpe(method="scirc", tau_min=0.2, tau_max=1000, N=40, n_psis=500, max_iter=1000, stop=False)
        # expe_tau = TauExpe(method="paral", tau_min=0.01, tau_max=200, max_iter=1000)
        expe_tau.whole(save=True)
        # expe_tau = TauExpe(method="scirc", tau_min=0.2, tau_max=1000, max_iter=1000)
        # expe_tau.whole(save=True)
    
    elif test_name == "tau_decl":
        expe_tau_dec_lagran = TauDecLagranExpe(method="paral", N=40, n_psis=500, max_iter=1000)
        expe_tau_dec_lagran.whole(save=True)
    
    elif test_name == "tau_dece":
        expe_tau_dec_euler = TauDecEulerExpe(N1=80, N2=25, n_psis=500, max_iter=30)
        expe_tau_dec_euler.whole(save=True)
    
    elif test_name == "convergence": 
        # Comparison of convergence on the basic case of 2 vMF shifted by an angle of pi/2, 
        # and with kappa = 100
        expe_convergence = ConvergenceExpe(N=50, n_psis=500, max_iter=1000, tau=[20, 50])
        expe_convergence.whole(save=True)

    elif test_name == "time":
        expe_time = TimeExpe(N_def=40, N_max=5000, n_psis_def=200, n_psis_max=5000, max_iter=20)
        expe_time.whole(save=True)

    elif test_name == "shape":
        expe_shape = ShapeExpe(N=200, n_psis=500, max_iter=1000, tau=[40, 80])
        expe_shape.whole(save=True)
    
    elif test_name == "epsb":
        expe_epsb = EPSBExpe(inputs="SmileyVMF", N1=150, N2=50, n_psis=100, tau=5e-3, max_iter=500, a_tau=0.05, pow_tau=1/2)
        expe_epsb.whole(save=True, representation = "3d")

    elif test_name == "benchmark":
        expe_benchmark = BenchmarkExpe(inputs="SmileyVMF", N=200, N1=150, N2=50, n_psis=100, reg=5e-2)
        expe_benchmark.whole(save=True, projection="3d")

    else:
        print("Test name not recognised")