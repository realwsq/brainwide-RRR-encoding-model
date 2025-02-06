import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import os, pickle

"""
helping functions
"""
def remove_space(s):
    s = s.replace(" ","")
    s = s.replace("'", "")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("{", "")
    s = s.replace("}", "")
    s = s.replace(":", "")
    s = s.replace(",", "_")
    return s

def log_kv(**kwargs):
    print(f"{kwargs}")

def make_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder

def load_or_save_dict(_fname, _main, **params):
    if os.path.isfile(_fname):
        with open(_fname, 'rb') as f:
            res = pickle.load(f)
    else:
        res = _main(**params)
        with open(_fname, 'wb') as f:
            pickle.dump(res, f)
    return res

# * cells should not contain nparray
def load_or_save_df(_fname, _main, **params):
    if os.path.isfile(_fname):
        df = pd.read_csv(_fname)
    else:
        df = _main(**params)
        df.to_csv(_fname, index=False,)
    return df

def create_dict(default, provided):
    result_dict = default.copy()
    result_dict.update(provided)
    return result_dict

def get_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device

def np2tensor(v):
    return torch.from_numpy(v)

def np2param(v, grad=True):
    return nn.Parameter(np2tensor(v), requires_grad=grad)

def tensor2np(v):
    return v.cpu().detach().numpy()



### compute R2
def compute_R2_main(y, y_pred, clip=True):
    """
    :y: (K, T, N) or (K*T, N)
    :y_pred: (K, T, N) or (K*T, N)
    """
    N = y.shape[-1]
    if len(y.shape) > 2:
        y = y.reshape((-1, N))
    if len(y_pred.shape) > 2:
        y_pred = y_pred.reshape((-1, N))
    r2s = np.asarray([r2_score(y[:, n], y_pred[:, n]) for n in range(N)])
    if clip:
        return np.clip(r2s, 0., 1.)
    else:
        return r2s
