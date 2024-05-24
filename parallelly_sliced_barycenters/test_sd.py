
import torch
torch.set_default_device('cpu')
from math import sqrt
from torch import pi

import matplotlib.pyplot as plt
import time

from src.sphere import Sphere, Sphere2, Sphere3, ScaSphere, ScaSphere3, VMFSphere
from src.barycenters import LSWBarycenter, LWBarycenter, wdist
from src.ssb import SSWBarycenter

from test import Test, logspace2_man
from utils import tqdm, USE_TQDM, repr2
from platform import node as platform_node
import cpuinfo

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Linux', call_pdb=False)

def validation_expe(d=5, N=50, n_psis=500, max_iter=100, tau=0.5):
    """Computes the LPSBarycenter of the given measures on the sphere, plot it and displays the loss evolution"""
    sca_sphere = ScaSphere(d=d)
    swb = LSWBarycenter(sca_sphere)
    
    X0 = sca_sphere.sample_uniform(N)
    coords = torch.tensor([[1.] + [0.]*(d-1)])
    kappas = torch.tensor([100.])
    Y = sca_sphere.sample_vMF(coords, kappas, N)
    lambdas = torch.Tensor([1.])

    swb.number_save = max_iter
    swb.fit(Y, lambdas, X0, n_psis, tau, max_iter)

    L_wdist = []
    for X in swb.L:
        L_wdist.append(wdist(X, Y[0]))

    plt.plot(swb.L_loss)
    plt.show()

    plt.plot(L_wdist)
    plt.show()

def generate_measures_set(distri, N):
    """Generates samples for a given setting of input measures.

    Args:
        distri (str): the setting to be generated. Among "distant vmf", "close vmf".
        N (int | tuple): Number of samples. If tuple, must have form (Nx, [Ny0, Ny1]) 
            with Nx the number of samples in the barycentre, Ny0 the number of samples in the 
            first input measure and Ny1 the number of samples in the second input measure.

    Returns:
        Tensor | list: the input measures. Is Tensor iff N is int. Else, returns a list with the two input measures (as samples) as Tensor.
    """
    # N: int or tuple (Nx, list [Nx_j for j])
    sphere3 = Sphere3()
    gnrl = type(N)== tuple
    if gnrl:
        Nx = N[0]
        Nys = N[1]
    else:
        Nx = N
        Nys = [N, N]
        
    if distri == "distant vmf":
        coords = sphere3.parametrisation(torch.tensor([[3*pi/4, 4*pi/8, 5*pi/8], [6*pi/4, 5*pi/8, 3*pi/8]]))
        if gnrl:
            Y = [sphere3.sample_vMF(coords[j], 100., Nys[j]) for j in range(2)]
        else:
            Y = sphere3.sample_vMF(coords, 100., N)
    
    elif distri == "close vmf":
        coords = sphere3.parametrisation(torch.tensor([[3*pi/4, 4*pi/8, pi/2], [5*pi/4, 5*pi/8, pi/2]]))
        if gnrl:
            Y = [sphere3.sample_vMF(coords[j], 300., Nys[j]) for j in range(2)]
        else:
            Y = sphere3.sample_vMF(coords, 300., N)

    return Y

def test_barycenter2(input_measures = "distant vmf", N=50, n_psis = 200, max_iter=1000, tau=1., compare_w=False, **kwargs):
    """Computes the barycentre of two given input measures (among a restricted set of possibilities), plot the results,
    and the convergence graph.

    Args:
        input_measures (str, optional): Input setting, among the different possible parameters of generate_measure_set. Defaults to "distant vmf".
        N (int, optional): Number of samples. If a tuple, it must be of the form (Nx, [Ny_j for j]) with Nx the number of sample in the barycentre,
            and Ny_j the number of sample in each input measure j. Defaults to 50.
        n_Qs (int, optional): Number of slices. Defaults to 200.
        max_iter (int, optional): Maximum number of iteration in the gradient descent algorithm. Defaults to 1000.
        tau (float, optional): Step size of the gradient descent algorithm. Defaults to 1..
        compare_w (bool, optional): Whether to compare the shape with the true Wasserstein barycenter. Default to False
    """
    # N: int or tuple (Nx, list [Nx_j for j])
    gnrl = type(N)== tuple
    if gnrl:
        Nx = N[0]
    else:
        Nx = N
    
    sman = ScaSphere3()
    psb = LSWBarycenter(sman)
    ssb = SSWBarycenter(sman)

    Y = generate_measures_set(input_measures, N)
    M = len(Y)

    X0 = sman.sample_uniform(Nx)
    sman.plot_samples(Y, X0, title="Initialisation")

    psb.fit(Y, None, X0, n_psis=n_psis, tau_init=tau, max_iter=max_iter, **kwargs)
    psb.plot_samples(title="Parallely sliced Wasserstein barycenter")

    ssb.fit(Y, None, X0, n_psis=n_psis, tau_init=tau*2, max_iter=max_iter)
    ssb.plot_samples(title="Semicircular sliced Wasserstein barycenter")

    if compare_w:
        wb = LWBarycenter(sman)
        wb.fit(Y, None)
        wb.plot_samples(title="Wasserstein barycenter")

    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(psb.L_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss PSB")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(psb.L_step)
    plt.xlabel("Iteration")
    plt.ylabel("Step diff")
    plt.title("Step diff PSB")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(ssb.L_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss SSB")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(ssb.L_step)
    plt.xlabel("Iteration")
    plt.ylabel("Step diff")
    plt.title("Step diff SSB")
    plt.grid()

    fig.tight_layout()
    plt.show()

###################################################################################


class ConvergenceExpe(Test):
    """Compare the convergence of LPSB and SSB algorithms for the case of 2vMF 
    - with centers Phi(pi/4, pi/2), Phi(-pi/4, pi/2)
    - and kappa=100
    """
    def __init__(self, d_list=[3, 100, 500, 1000], N=50, n_psis=500, max_iter=1000, tau=[20, 50], repetitions=5):
        """
        Args:
            d_list (list, optional): List of integer. Ambiant dimensions to test. Defaults to [3, 100, 500, 1000].
            N (int | tuple, optional): Number of points in the distribution. If tuple, first element is the number
                of points in the barycentre and second element is a list [Ny_j for j] with the number of points of 
                each input measure. Defaults to 50.
            n_psis (int, optional): Number of slices. Defaults to 500.
            max_iter (int, optional): Maximum number of iterations in the gradient descent algorithms. Defaults to 1000.
            tau (int | list, optional): Step size of the gragient descent algorithms. If list, the first element is 
                the step size for the PSB algorithm and the second is the step size for the SSB algorithm. 
                Defaults to [20, 50].
            repetitions (int, optional): Number of repetition of each case. We then consider the mean and the 
                standard deviation. Defaults to 5.
        """
        super().__init__()
        self.d_list      = d_list
        self.N           = N
        self.n_psis      = n_psis
        self.max_iter    = max_iter
        self.repetitions = repetitions
        if type(tau) != list:
            self.tau = [tau, tau]
        else:
            self.tau = tau

        self.measures = None
        self.X0 = None
        # self.pbarycenter = None
        self.pL_loss = torch.zeros((len(d_list), max_iter, repetitions))
        self.pL_step = torch.zeros((len(d_list), max_iter, repetitions))
        # self.sbarycenter = None
        self.sL_loss = torch.zeros((len(d_list), max_iter, repetitions))
        self.sL_step = torch.zeros((len(d_list), max_iter, repetitions))
        
    
    @property
    def name(self):
        return f"expe_cvrg dim_N{self.N}_n{self.n_psis}_mi{self.max_iter}_tau{self.tau[0]}-{self.tau[1]}_{self.repetitions}runs"

    def run(self):
        self.node = platform_node()

        # rewriting input parameters
        gnrl = type(self.N)== tuple #whether to used the generalised algorithm version
        if gnrl:
            Nx = self.N[0]
            Nys = self.N[1]
        else:
            Nx = self.N
        
        for i, d in enumerate(tqdm(self.d_list)):
            sphere = Sphere(d=d)
            # defining input measures
            coords = torch.tensor([[1, 1] + [0]*(d-2), [1, -1] + [0]*(d-2)], dtype=torch.float)/sqrt(2)
            kappas = torch.tensor([100, 100], dtype=torch.float)
            if gnrl:
                self.measures = [sphere.sample_vMF(coords[j], kappas[j], Nys[j]) for j in range(2)]
            else:
                self.measures = sphere.sample_vMF(coords, kappas, self.N)

            for k in tqdm(range(self.repetitions), leave=False):
                # defining initialisation
                self.X0 = sphere.sample_uniform(Nx)

                # running experiments
                psb = LSWBarycenter(ScaSphere(d=d))
                ssb = SSWBarycenter()
                psb.fit(self.measures, None, self.X0, n_psis=self.n_psis, tau_init=self.tau[0], max_iter=self.max_iter, stop=False, tqdm_leave=False)
                ssb.fit(self.measures, None, self.X0, n_psis=self.n_psis, tau_init=self.tau[1], max_iter=self.max_iter, stop=False, tqdm_leave=False)

                # saving results
                # self.pbarycenter = psb.barycenter
                self.pL_loss[i,:,k] = torch.tensor(psb.L_loss)
                self.pL_step[i,:,k] = torch.tensor(psb.L_step)
                # self.sbarycenter = ssb.barycenter
                self.sL_loss[i,:,k] = torch.tensor(ssb.L_loss)
                self.sL_step[i,:,k] = torch.tensor(ssb.L_step)

        self.completed = True
    
    def plot(self):
        # print parameters
        for k in ["node", "d_list", "N", "n_psis", "max_iter", "tau", "repetitions"]:
            print(f"{k:8}: {repr2(self.__getattribute__(k))}")

        plt.figure(self.name + "_loss")
        for i, d in enumerate(self.d_list):
            #plt.plot(torch.mean(self.pL_loss[i], dim=-1), label="Parallel")
            #plt.plot(8.5*torch.mean(self.sL_loss[i], dim=-1), label="Semi-circular")
            pL_mean = torch.mean(self.pL_loss[i],dim=[0,1])
            errorbar_container = plt.errorbar(range(self.max_iter), torch.mean(self.pL_loss[i]/pL_mean, dim=-1), torch.std(self.pL_loss[i]/pL_mean, dim=-1), linestyle='None', marker='d', label="Parallel")#label=f"d={d}")
            colour = errorbar_container.lines[0].get_c()
            sL_mean = torch.mean(self.sL_loss[i],dim=[0,1])
            plt.errorbar(range(self.max_iter), torch.mean(self.sL_loss[i]/sL_mean, dim=-1), torch.std(self.sL_loss[i]/sL_mean, dim=-1), linestyle='None', marker='d', label="Semi-circular")#, c=colour)
        plt.legend()
        # plt.title("(Rescaled) Loss evolution over the iterations")
        plt.grid()
        plt.savefig(self.name + "_loss.png")
        plt.show()

        plt.figure(self.name + "_step")
        for i, d in enumerate(self.d_list):
            #plt.plot(torch.mean(self.pL_step[i], dim=-1), label="Parallel")
            #plt.plot(torch.mean(self.sL_step[i], dim=-1), label="Semi-circular")
            errorbar_container = plt.errorbar(range(self.max_iter), torch.mean(self.pL_step[i], dim=-1), torch.std(self.pL_step[i], dim=-1), linestyle='None', marker='d', label="Parallel")#label=f"d={d}")
            colour = errorbar_container.lines[0].get_c()
            plt.errorbar(range(self.max_iter), torch.mean(self.sL_step[i], dim=-1), torch.std(self.sL_step[i], dim=-1), linestyle='None', marker='d', label="Semi-circular")#, c=colour)
        plt.legend()
        # plt.title("Step norm for each iteration")
        plt.grid()
        plt.savefig(self.name + "_step.png")
        plt.show()



class TimeExpe(Test):
    """Compare the execution times of the LPSB and SSB algorithms with the same number of iterations, for the case of 2 vMF distributions,
    in terms of influence of the number N of points and the number n_psi of slices."""
    def __init__(self, d_list = [3, 100, 500, 1000], N_def=40, N_max=100, n_psis_def=200, n_psis_max=500, max_iter=100, repetitions=1):
        super().__init__()
        self.d_list = d_list
        self.N_def = N_def
        self.N_max = N_max
        self.n_psis_def = n_psis_def
        self.n_psis_max = n_psis_max
        self.max_iter = max_iter
        self.tau = 20
        self.repetitions = repetitions

        self.N_values = logspace2_man(N_max*2)
        self.n_psis_values = logspace2_man(n_psis_max*2)
        self.times_N = torch.zeros((len(d_list), len(self.N_values), 2, self.repetitions))
        self.times_n_psis = torch.zeros((len(d_list), len(self.n_psis_values), 2, self.repetitions))
    
    @property
    def name(self):
        return f"expe_time_dim_N{self.N_def}-{self.N_max}_n{self.n_psis_def}-{self.n_psis_max}_mi{self.max_iter}_{self.repetitions}runs"
    
    def run(self):
        self.node = platform_node()
        self.cpu =  cpuinfo.get_cpu_info()['brand_raw']

        for i, d in enumerate(tqdm(self.d_list)):
            sphere = Sphere(d=d)
            ssb = SSWBarycenter()
            psb = LSWBarycenter(ScaSphere(d=d))

            coords = torch.tensor([[1, 1] + [0]*(d-2), [1, -1] + [0]*(d-2)], dtype=torch.float)/sqrt(2)
            kappas = torch.tensor([100, 100], dtype=torch.float)

            for r in tqdm(range(self.repetitions), leave=False):
                for j, N in enumerate(tqdm(self.N_values, leave=False)):
                    X0 = sphere.sample_uniform(N)
                    Y = sphere.sample_vMF(coords, kappas, N)
    
                    t0 = time.time()
                    psb.fit(Y, None, X0, n_psis=self.n_psis_def, tau_init=self.tau, max_iter=self.max_iter, tqdm_leave=False, stop=False)
                    t1 = time.time()
                    self.times_N[i, j, 0, r] = t1 - t0
    
                    t0 = time.time()
                    ssb.fit(Y, None, X0, n_psis=self.n_psis_def, tau_init=self.tau, max_iter=self.max_iter, tqdm_leave=False, stop=False)
                    t1 = time.time()
                    self.times_N[i, j, 1, r] = t1 - t0
            
                X0 = sphere.sample_uniform(self.N_def)
                Y = sphere.sample_vMF(coords, kappas, self.N_def)
                for j, n_psis in enumerate(tqdm(self.n_psis_values, leave=False)):
    
                    t0 = time.time()
                    psb.fit(Y, None, X0, n_psis=n_psis, tau_init=self.tau, max_iter=self.max_iter, tqdm_leave=False, stop=False)
                    t1 = time.time()
                    self.times_n_psis[i, j, 0, r] = t1 - t0
    
                    t0 = time.time()
                    ssb.fit(Y, None, X0, n_psis=n_psis, tau_init=self.tau, max_iter=self.max_iter, tqdm_leave=False, stop=False)
                    t1 = time.time()
                    self.times_n_psis[i, j, 1, r] = t1 - t0
                    # Y = torch.stack(Y)
        self.completed = True
    
    def plot(self):
        # print parameters
        for k in ["node", "cpu", "N_def", "N_max", "n_psis_def", "n_psis_max", "max_iter", "tau", "repetitions"]:
            print(f"{k:10}: {repr(self.__getattribute__(k))}")
        
        # plot results
        plt.figure(f"{self.name}_wrt_N")
        for i, d in enumerate(self.d_list):
            t_N_mean = self.times_N[i].mean(dim=2).cpu()
            line, = plt.loglog(self.N_values.cpu(), t_N_mean[:, 0], label=f"d={d}")
            plt.loglog(self.N_values.cpu(), t_N_mean[:, 1], "--", c=line.get_c())
        # plt.loglog(self.N_values, 0.03*self.N_values)
        plt.legend()
        plt.grid(True, which="both")
        plt.xlabel("N")
        plt.ylabel("Time (s)")
        plt.show()

        plt.figure(f"{self.name}_wrt_npsi")
        for i, d in enumerate(self.d_list):
            t_psis_mean = self.times_n_psis[i].mean(dim=2).cpu()
            line, = plt.loglog(self.n_psis_values.cpu(), t_psis_mean[:, 0], label=f"d={d}")
            plt.loglog(self.n_psis_values.cpu(), t_psis_mean[:, 1], "--", c=line.get_c())
        plt.legend()
        plt.grid(True, which="both")
        plt.xlabel("P")
        plt.ylabel("Time (s)")
        plt.show()

class ShapeExpe(Test):
    """Compare the shapes of the LPSBarycenter and SSBarycenter for different input settings."""
    def __init__(self, N=100, n_psis=500, max_iter=1000, tau=[40, 80]):
        super().__init__()
        self.N = N
        # N: int or tuple (Nx, list [Nx_j for j])
        self.n_psis=n_psis
        self.max_iter=max_iter
        if type(tau) == list:
            self.tau = tau
        else:
            self.tau = [tau, tau]
        
        sphere3 = Sphere3()
        coords_distant = sphere3.parametrisation(torch.tensor([[3*pi/4, 4*pi/8, 5*pi/8], [6*pi/4, 5*pi/8, 3*pi/8]]))
        coords_close = sphere3.parametrisation(torch.tensor([[3*pi/4, 4*pi/8, pi/2], [5*pi/4, 5*pi/8, pi/2]]))
        self.input_params = {
            "2vmf_distant" : [["vMF", coords_distant[0], 100], ["vMF", coords_distant[1], 100]],
            "2vmf_close"   : [["vMF", coords_close[0], 300], ["vMF", coords_close[1], 300]],
        }

        self.Y_list = {}
        # self.X_paral_list = []
        # self.X_scirc_list = []
        self.X_list = {"PSB":{}, "SSB":{}, "WB":{}}
        self.times  = {"PSB":{}, "SSB":{}, "WB":{}}
        self.L_loss = {"PSB":{}, "SSB":{}}
        self.X0 = None
    
    @property
    def name(self):
        return f"expe_shape_s3_N{self.N}_n{self.n_psis}_mi{self.max_iter}_tau{self.tau[0]}-{self.tau[1]}"

    def run(self):
        self.node = platform_node()
        gnrl = type(self.N)== tuple
        if gnrl:
            Nx = self.N[0]
            Nys = self.N[1]
        else:
            Nx = self.N
            Nys = [self.N, self.N]

        param_to_measure = lambda param: VMFSphere(*param[1:])
        
        sphere = Sphere3()
        self.X0 = sphere.sample_uniform(Nx)
        # for m1, m2 in measures_pairs:
        for name, (input1, input2) in self.input_params.items():
            m1 = param_to_measure(input1)
            m2 = param_to_measure(input2)
            Y = [m1.sample(Nys[0]), m2.sample(Nys[1])]
            if not gnrl:
                Y = torch.stack(Y)
            self.Y_list[name] = Y

            psb = LSWBarycenter(ScaSphere3())
            ssb = SSWBarycenter()
            wb = LWBarycenter(sphere)

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

            t0 = time.time()
            wb.fit(Y, None)
            self.times["WB"][name] = time.time() - t0
            self.X_list["WB"][name] = wb.barycenter

        self.completed = True
    
    def plot(self):
        sphere = Sphere3()

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
            
            sphere.plot_samples(self.Y_list[input_name], self.X_list["WB"][input_name], figname=f"{self.name}_{input_name}_WB")


#####################################################################################

if __name__ == "__main__":
    # sca_sphere = ScaSphere(d=)

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-t", "--test-name", type=str, default="visual",
                        help="Name of test to launch. Can be 'visual', 'tau', 'tau_decl', 'tau_dece', 'convergence', 'time', 'shape', 'epsb', 'benchmark'. Defaults to 'visual'.")
    parser.add_argument("-w", "--without-tqdm", action="store_true",
                        help="Prevent tqdm bars")
    args = parser.parse_args()
    test_name = args.test_name
    USE_TQDM[0] = not args.without_tqdm

    if test_name == "validation":
        validation_expe(d=5, N=50, n_psis=200, max_iter=1000, tau=10.)
    
    elif test_name == "convergence":
        #expe_cvrg = ConvergenceExpe(d_list=[3, 100, 500, 1000], N=20, n_psis=100, max_iter=80, tau=[20, 50], repetitions=10)
        expe_cvrg = ConvergenceExpe(d_list=[10], N=20, n_psis=100, max_iter=80, tau=[20, 50], repetitions=100)
        expe_cvrg.whole(save=True)

    elif test_name == "time":
        # expe_time = TimeExpe(N_def=40, N_max=5000, n_psis_def=200, n_psis_max=5000, max_iter=20, repetitions=5) #article
        expe_time = TimeExpe(N_def=40, N_max=5000, n_psis_def=100, n_psis_max=5000, max_iter=10, repetitions=5)
        expe_time.whole(save=True)
    
    elif test_name == "shape":
        # test_barycenter2(input_measures="distant vmf", N=100, n_psis=300, max_iter=1000, tau=50., compare_w=True)
        # test_barycenter2(input_measures="close vmf"  , N=100, n_psis=300, max_iter=1000, tau=50., compare_w=True)
        expe_shape = ShapeExpe(N=100, n_psis=300, max_iter=1000, tau=[40, 80])
        expe_shape.whole(save=True)

    else:
        print("Test name not recognised")
