import torch
from math import log10, floor

from abc import abstractmethod
import os
import pickle


class Test():
    """Inheriting classes must redefine 
        - name(self), 
        - run(self), 
        - plot(self), 
        - and integrate command `self.completed = True` in run(self).
    """
    def __init__(self):
        self.completed = False
        self.res_folder = "res_test/"
        self.node = None

    @property
    @abstractmethod
    def name(self):
        return "None"
    
    @property
    def path(self):
        return os.path.join(self.res_folder, self.name + ".pkl")
    
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    def save(self):
        if not os.path.exists(self.res_folder):
            os.makedirs(self.res_folder)
        with open(self.path, "wb") as file:
            pickle.dump(self.__dict__, file)
        
    def load(self):
        with open(self.path, "rb") as file:
            saved_dict = pickle.load(file)
            self.__dict__.update(saved_dict)

    def whole(self, save=False, **kwargs):
        if not self.completed:
            if os.path.isfile(self.path):
                self.load()
            else:
                self.run()
                if save:
                    self.save()
        self.plot(**kwargs)

def logspace2_man(a, b=None):
    if b is None:
        deb = 1
        fin = a
    else:
        deb = a
        fin = b
    m_values = torch.tensor([1, 2, 5])

    p_min = floor(log10(deb))
    p_max = floor(log10(fin)) + 1
    values = torch.cat([m_values*10**p for p in range(p_min, p_max)])
    values = values[(values >= deb) & (values < fin)]
    return values
