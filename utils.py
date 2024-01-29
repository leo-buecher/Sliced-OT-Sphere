from tqdm import tqdm as basetqdm
from math import log2, log10
import time

USE_TQDM = [True] #! change this

def unsqueeze(tensor, dim, n=1):
    """Homemade equivalent for torch.unsqueeze. The difference is that we can do a multiple unsqueezes in a row.

    Args:
        tensor (Tensor): Tensor to unsqueeze.
        dim (int): Dimension to unsqueeze (between -tensor.ndim-1 (included) and tensor.ndim+1 (excluded))
            If negative, same as tensor.ndim + 1 + dim.
        n (int, optional): Number of times to unsqueeze the dimension. Defaults to 1.

    Returns:
        Tensor: Unsqueezed tensor.
    """
    if dim < 0:
        dim = tensor.ndim + 1 + dim
    return tensor[(slice(None),)*dim + (None,)*n]

def hm_tqdm(iterator):
    if hasattr(iterator, "__len__"):
        length = len(iterator)
        d = 1 + int(log10(length))
    else:
        length = None

    # Loop
    t0 = time.time()
    for i, x in enumerate(iterator):
        if i == 0:
            if length is None:                    
                print(f"it {i:4}")
            else:
                print(f"it {i:{d}} / {length}")
        elif log2(i)%1 == 0: # Progress is reported at iterations 1, 2, 4, 8, ...
            t1 = time.time()
            if length is None:
                print(f"it {i:4}, exec time: {t1 - t0}s")
            else:
                estim = (t1-t0)*length/i
                print(f"it {i:{d}} / {length}, exec time: {t1 - t0}s / {estim}s")
        yield x

    # End
    t1 = time.time()
    if length is None:
        print(f"it {i + 1:4}, total exec time: {t1 - t0}s")
    else:
        estim = (t1-t0)*length/i
        print(f"it {i + 1:{d}} / {length}, total exec time: {t1 - t0}s")
    print()

def tqdm(iterator, leave=True):
    if USE_TQDM[0]:
        return basetqdm(iterator, leave=leave)
    else:
        return hm_tqdm(iterator)

def repr2(dico, indent="   ", depth=1):
    if type(dico) == dict :
        r = "{\n"
        for k, v in dico.items():
            repr_v = repr2(v, indent=indent, depth=depth+1)
            r+= indent*depth + repr(k) + ": " + repr_v + ",\n"
        r += indent*(depth-1) + "}"
        return(r)
    else :
        return repr(dico)