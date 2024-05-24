import torch
import matplotlib.pyplot as plt
from torch import pi

import time
from utils import tqdm, USE_TQDM

from src.so3 import SO3, ScaSO3, DistSO3
from src.barycenters import LSWBarycenter, LWBarycenter
from test import Test, logspace2_man

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

def expe(Y, n_Qs=1000, max_iter=1000, X0=None, lambdas=None, tau=0.5):
    sca_so3 = ScaSO3()
    swb = LSWBarycenter(sca_so3)
    if X0 is None:
        X0 = sca_so3.sample_uniform(Y.shape[1])
    
    sca_so3.plot_samples(Y, X0)

    swb.fit(Y, lambdas, X0, n_psis=n_Qs, tau_init=tau, max_iter=max_iter)
    swb.plot_samples()

    plt.plot(swb.L_loss)
    plt.show()

def generate_measures_set(distri, N):
    """Generates samples for a given setting of input measures.

    Args:
        distri (str): the setting to be generated. Among "projected gaussains", "lines", 
            "cuboids", "distant vmf", "close vmf", "line vmf".
        N (int | tuple): Number of samples. If tuple, must have form (Nx, [Ny0, Ny1]) 
            with Nx the number of samples in the barycentre, Ny0 the number of samples in the 
            first input measure and Ny1 the number of samples in the second input measure.

    Returns:
        Tensor | list: the input measures. Is Tensor iff N is int. Else, returns a list with the two input measures (as samples) as Tensor.
    """
    # N: int or tuple (Nx, list [Nx_j for j])
    so3 = SO3()
    gnrl = type(N)== tuple
    if gnrl:
        Nx = N[0]
        Nys = N[1]
    else:
        Nx = N
        Nys = [N, N]

    #TODO: GENERALISE
    if distri == "projected gaussians":
        R0 = torch.eye(3)
        th = torch.tensor(pi/4)
        R1 = torch.tensor([[1, 0, 0], [0, torch.cos(th), -torch.sin(th)], [0, torch.sin(th), torch.cos(th)]])
        coords = torch.stack([R0, R1])
        stds = torch.tensor([0.1, 0.1])
        if gnrl:
            Y = [so3.sample_projected_gaussian(coords[j], stds[j], Nys[j]) for j in range(2)]
        else:
            Y = so3.sample_projected_gaussian(coords, stds, N)

    elif distri == "lines": # two lines rotated by an angle of pi/2
        Y0 = so3.sample_uniform_portion(torch.tensor([[3/8*pi, 5/8*pi], [pi/2, pi/2], [pi, pi]]), Nys[0])
        Y1 = so3.sample_uniform_portion(torch.tensor([[7/8*pi, 9/8*pi], [pi/2, pi/2], [pi, pi]]), Nys[1])
        if gnrl:
            Y = [Y0, Y1]
        else:
            Y = torch.stack([Y0, Y1])
        # X_cand = so3.sample_uniform_portion(torch.tensor([[5/8*pi, 7/8*pi], [pi/2, pi/2], [pi, pi]]), Nx)

    elif distri == "cuboids": # two cuboids rotated by an angle of pi/2
        # Fourth attempt
        Y0 = so3.sample_uniform_portion(torch.tensor([[23/16*pi, 25/16*pi], [pi*3/8, pi*5/8], [pi*3/4, pi*5/4]]), Nys[0])
        Y1 = so3.sample_uniform_portion(torch.tensor([[31/16*pi, 33/16*pi], [pi*3/8, pi*5/8], [pi*3/4, pi*5/4]]), Nys[0])
        if gnrl:
            Y = [Y0, Y1]
        else:
            Y = torch.stack([Y0, Y1])
        # X_cand = so3.sample_uniform_portion(torch.tensor([[27/16*pi, 29/16*pi], [pi*3/8, pi*5/8], [pi*3/4, pi*5/4]]), Nx)
        
    elif distri == "distant vmf":
        coords = so3.parametrisation(torch.tensor([[3*pi/4, 6*pi/4], [4*pi/8, 5*pi/8], [5*pi/4, 3*pi/4]]))
        dist = torch.acos((torch.trace(torch.matmul(coords[0,:,:].transpose(0,1),coords[1,:,:])) - 1) / 2)
        print(distri, dist.item())
        # TODO: change so that batch size comes first, to be coherent with sphere.
        if gnrl:
            Y = [so3.sample_vMF_quat(coords[j], 100., Nys[j]) for j in range(2)]
        else:
            Y = so3.sample_vMF_quat(coords, 100., N)
        #X_cand = so3.sample_vMF_quat(so3.parametrisation(torch.tensor([9*pi/8, 9*pi/16, pi/2])), 100., Nx)
    
    elif distri == "close vmf":
        coords = so3.parametrisation(torch.tensor([[3*pi/4, 5*pi/4], [4*pi/8, 5*pi/8], [pi, pi]]))
        dist = torch.acos((torch.trace(torch.matmul(coords[0,:,:].transpose(0,1),coords[1,:,:])) - 1) / 2)
        print(distri, dist.item())
        # TODO: change so that batch size comes first, to be coherent with sphere.
        if gnrl:
            Y = [so3.sample_vMF_quat(coords[j], 300., Nys[j]) for j in range(2)]
        else:
            Y = so3.sample_vMF_quat(coords, 300., N)
    
    elif distri == "line vmf":
        Y0 = so3.sample_uniform_portion(torch.tensor([[0, 2*pi], [2*pi/8, 4*pi/8], [6*pi/8, 8*pi/8]]), N)
        # coords = so3.parametrisation(torch.tensor([pi, 6*pi/8, 12*pi/8]))
        coords = so3.parametrisation(torch.tensor([0, 5*pi/8, 14*pi/8]))
        Y1 = so3.sample_vMF_quat(coords, 300, N)
        if gnrl:
            Y = [Y0, Y1]
        else:
            Y = torch.stack([Y0, Y1])

    return Y


def compare_dist_sca(N=13, n_Qs = 200, max_iter=1000, eps=1e-1, **kwargs):
    sca_so3 = ScaSO3()
    swbs = LSWBarycenter(sca_so3)

    dist_so3 = DistSO3(eps=eps)
    swbd = LSWBarycenter(dist_so3)

    # Y, X_cand = generate_measures_set("distant vmf", N)
    Y = generate_measures_set("distant vmf", N)
    M = Y.shape[0]
    psis_cand = sca_so3.sample_uniform(n_Qs)
    # _, obj_cand = swbs.functional_grad(psis_cand, X_cand, Y, 1/M * torch.ones((M,)))

    X0 = sca_so3.sample_uniform(N)
    sca_so3.plot_samples(Y, X0)

    swbs.fit(Y, None, X0, n_psis=n_Qs, tau_init=1, max_iter=max_iter, **kwargs)
    swbs.plot_samples()

    # _, obj_cand2 = swbd.functional_grad(psis_cand, X_cand, Y, 1/M * torch.ones((M,)))
    swbd.fit(Y, None, X0, n_psis=n_Qs, tau_init=1, max_iter=max_iter, **kwargs)
    swbs.plot_samples()

    plt.plot(swbs.L_loss)
    plt.plot(swbd.it_save_range, [swbs.functional_grad(psis_cand, X, Y, torch.ones((M,))/M)[1] for X in swbd.L])
    for i in range(50):
        plt.axhline(1/4 * swbs.sliced_dist_square(sca_so3.sample_uniform(n_Qs), Y[0], Y[1]), c="r", lw=0.2)
    # plt.axhline(obj_cand, c="g")
    plt.show()

    plt.plot(swbs.it_save_range, [swbd.functional_grad(psis_cand, X, Y, torch.ones((M,))/M)[1] for X in swbs.L])
    plt.plot(swbd.L_loss)
    for i in range(50):
        plt.axhline(1/4 * swbd.sliced_dist_square(sca_so3.sample_uniform(n_Qs), Y[0], Y[1]), c="r", ls="--", lw=0.2)
    # plt.axhline(obj_cand2, c="g", ls="--")
    plt.show()

    plt.plot(swbs.L_step)
    plt.plot(swbd.L_step)
    plt.show()
    
def test_barycenter2(input_measures = "distant vmf", N=50, n_psis = 200, max_iter=1000, slicing_operator_arcos=False, tau=1., eps=1e-1, compare_w=False, **kwargs):
    """Computes the barycentre of two given input measures (among a restricted set of possibilities), plot the results,
    and the convergence graph.

    Args:
        input_measures (str, optional): Input setting, among the different possible parameters of generate_measure_set. Defaults to "distant vmf".
        N (int, optional): Number of samples. If a tuple, it must be of the form (Nx, [Ny_j for j]) with Nx the number of sample in the barycentre,
            and Ny_j the number of sample in each input measure j. Defaults to 50.
        n_psis (int, optional): Number of slices. Defaults to 200.
        max_iter (int, optional): Maximum number of iteration in the gradient descent algorithm. Defaults to 1000.
        slicing_operator_arcos (bool, optional): Whether to use the distance on SO3 as slicing operator, or simply the scalar product. 
            Defaults to False.
        tau (float, optional): Step size of the gradient descent algorithm. Defaults to 1..
        eps (float, optional): Parameter epsilon used for the slicing operator with distance on SO3. Ignored if slicing_operator_arcos is False. 
            Defaults to 1e-1.
        compare_w (bool, optional): Whether to compare the shape with the true Wasserstein barycenter. Default to False
    """
    # N: int or tuple (Nx, list [Nx_j for j])
    gnrl = type(N)== tuple
    if gnrl:
        Nx = N[0]
    else:
        Nx = N
    
    if slicing_operator_arcos:
        sman = DistSO3(eps=eps)
    else:
        sman = ScaSO3()
    swb = LSWBarycenter(sman)

    Y = generate_measures_set(input_measures, N)
    M = len(Y)

    X0 = sman.sample_uniform(Nx)
    sman.plot_samples(Y, X0, figname="so3 " + input_measures + " init") # title="Initialisation", 

    swb.fit(Y, None, X0, n_psis=n_psis, tau_init=tau, max_iter=max_iter, **kwargs)
    swb.plot_samples(figname="so3 " + input_measures + " sw") # title="Parallely sliced Wasserstein barycenter",

    if compare_w:
        wb = LWBarycenter(sman)
        wb.fit(Y, None)
        wb.plot_samples(figname="so3 " + input_measures + " w") #title="Wasserstein barycenter",

    # plt.plot(swb.L_loss)
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")
    # plt.grid()
    # plt.show()

    # plt.plot(swb.L_step)
    # plt.xlabel("Iteration")
    # plt.ylabel("Step diff")
    # plt.grid()
    # plt.show()


###################################################################################

class SO3SpeedTest(Test):
    def __init__(self, N_max=500, n_psis_max=500, max_iter=1000):
        super().__init__()
        self.N_max = N_max
        self.n_psis_max = n_psis_max
        self.max_iter = max_iter

        self.N_values = logspace2_man(N_max)
        self.n_psis_values = logspace2_man(n_psis_max)
        self.times = torch.zeros((len(self.N_values), len(self.n_psis_values), 3))


    @property
    def name(self):
        return f"test_so3_time_{self.N_max}_{self.n_psis_max}_{self.max_iter}"
    
    def run(self):
        sca_so3 = ScaSO3()
        swb = LSWBarycenter(sca_so3)

        R0 = torch.eye(3)
        th = torch.pi/4
        R1 = torch.tensor([[1, 0, 0], [0, torch.cos(th), -torch.sin(th)], [0, torch.sin(th), torch.cos(th)]])
        coords = torch.stack([R0, R1])
        stds = torch.tensor([0.1, 0.1])

        for i, N in tqdm(enumerate(self.N_values)):
            Y = sca_so3.sample_projected_gaussian(coords, stds, N)
            X0 = sca_so3.sample_projected_gaussian(torch.tensor([[0, 0, 1]], dtype=torch.float), torch.tensor([0.3]), N)[0]
            for j, n_Qs in enumerate(self.n_psis_values):
                t0 = time.time()
                swb.fit(Y, None, X0, n_psis=n_Qs, tau_init=0.5, max_iter=self.max_iter)
                t1 = time.time()
                self.times[i, j] = t1 - t0
        self.completed=True

    def plot(self):
        for i in range(self.times.shape[0]):
            plt.loglog(self.n_psis_values, self.times[i,:])
        plt.loglog(self.n_psis_values, self.n_psis_values/self.n_psis_values[0]*self.times[0,0], "k:")
        plt.grid(True, "both")
        plt.show()

class SO3AccuTest(Test):
    def __init__(self, N_max=500, n_psis_max=500, max_iter=1000):
        super().__init__()
        self.N_max = N_max
        self.n_psis_max = n_psis_max
        self.max_iter = max_iter

        self.N_values = logspace2_man(N_max)
        self.n_psis_values = logspace2_man(n_psis_max)
        self.obj = torch.zeros((len(self.N_values), len(self.n_psis_values), 2))
        self.dist = torch.zeros((len(self.N_values), len(self.n_psis_values), 2))

    @property
    def name(self):
        return f"test_so3_accu_{self.N_max}_{self.n_psis_max}_{self.max_iter}"

    def run(self):
        sca_so3 = ScaSO3()
        swb = LSWBarycenter(sca_so3)

        R0 = torch.eye(3)
        th = torch.pi/4
        R1 = torch.tensor([[1, 0, 0], [0, torch.cos(th), -torch.sin(th)], [0, torch.sin(th), torch.cos(th)]])
        coords = torch.stack([R0, R1])
        stds = torch.tensor([0.1, 0.1])

        for i, N in tqdm(enumerate(self.N_values)):
            Y = sca_so3.sample_projected_gaussian(coords, stds, N)
            X0 = sca_so3.sample_projected_gaussian(torch.tensor([[0, 0, 1]], dtype=torch.float), torch.tensor([0.3]), N)[0]
            for j, n_Qs in enumerate(self.n_psis_values):
                swb.fit(Y, None, X0, n_psis=n_Qs, tau_init=0.5, max_iter=self.max_iter)
                hist = torch.tensor(swb.L_loss)
                hist_end = hist[9*self.max_iter//10 :]
                self.obj[i, j, 0] = torch.mean(hist_end)
                self.obj[i, j, 1] = torch.std(hist_end)
                
                dist_samples = torch.zeros((self.max_iter//10,))
                for k in range(len(dist_samples)):
                    dist_samples[k] = swb.sliced_dist_square(sca_so3.sample_uniform(n_Qs), Y[0], Y[1])
                self.dist[i, j, 0] = torch.mean(dist_samples)
                self.dist[i, j, 1] = torch.std(dist_samples)
        self.completed = True
        
    def plot(self):
        for i in range(self.obj.shape[0]):
            plt.semilogx(self.n_psis_values, self.obj[i, :, 0], label=f"$N =$ {self.N_values[i]}")
            plt.fill_between(self.n_psis_values, self.obj[i, :, 0] + self.obj[i, :, 1], self.obj[i, :, 0] - self.obj[i, :, 1], alpha=0.5)
        plt.xlabel("$n_Q$")
        plt.ylabel("Objective value")
        plt.legend()
        plt.show()

        for j in range(self.obj.shape[1]):
            plt.semilogx(self.N_values, self.obj[:, j, 0], label=f"$n_Q =$ {self.n_psis_values[j]}")
            plt.fill_between(self.N_values, self.obj[:, j, 0] + self.obj[:, j, 1], self.obj[:, j, 0] - self.obj[:, j, 1], alpha=0.5)
        plt.xlabel("$N$")
        plt.ylabel("Objective value")
        plt.legend()
        plt.show()

        for i in range(self.dist.shape[0]):
            plt.semilogx(self.n_psis_values, self.dist[i, :, 0], label=f"$N =$ {self.N_values[i]}")
            plt.fill_between(self.n_psis_values, self.dist[i, :, 0] + self.dist[i, :, 1], self.dist[i, :, 0] - self.dist[i, :, 1], alpha=0.5)
        plt.xlabel("$n_Q$")
        plt.ylabel("Objective value")
        plt.legend()
        plt.show()

        for j in range(self.dist.shape[1]):
            plt.semilogx(self.N_values, self.dist[:, j, 0], label=f"$n_Q =$ {self.n_psis_values[j]}")
            plt.fill_between(self.N_values, self.dist[:, j, 0] + self.dist[:, j, 1], self.dist[:, j, 0] - self.dist[:, j, 1], alpha=0.5)
        plt.xlabel("$N$")
        plt.ylabel("Objective value")
        plt.legend()
        plt.show()

######################################################################################

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-t", "--test-name", type=str, default="visual",
                        help="Name of test to launch. Can be 'visual', 'tau', 'tau_decl', 'tau_dece', 'convergence', 'time', 'shape', 'epsb', 'benchmark'. Defaults to 'visual'.")
    parser.add_argument("-w", "--without-tqdm", action="store_true",
                        help="Prevent tqdm bars")
    args = parser.parse_args()
    test_name = args.test_name
    USE_TQDM[0] = not args.without_tqdm

    if test_name == "shape":
        test_barycenter2(input_measures="distant vmf", N=100, n_psis=500, max_iter=1000, slicing_operator_arcos=True, tau=50., compare_w=True)
        test_barycenter2(input_measures="close vmf"  , N=100, n_psis=500, max_iter=1000, slicing_operator_arcos=True, tau=50., compare_w=True)
        test_barycenter2(input_measures="line vmf"   , N=300, n_psis=500, max_iter=1000, slicing_operator_arcos=True, tau=50., compare_w=True)

    # elif test_name == "compare_slicing_operator":
    #     compare_dist_sca(N=30, n_Qs=200, max_iter=1000, eps=1e-1)


    else:
        print("Test name not recognised")


