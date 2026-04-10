import os
import numpy as np
from sklearn.metrics import r2_score

import torch
import torch.nn as nn

"""
helping functions
"""
def log_kv(**kwargs):
    print(f"{kwargs}")

def make_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder

def get_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. 
    # We'll use this device variable later in our code.
    if is_cuda:
        gpu_id = 0
        device = torch.device(f'cuda:{gpu_id}')
        print(f"GPU:{gpu_id} is available")
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

def p_to_text(p):
    if p < 0.0001:
        return '*** P=%.1e' % p
    if p < 0.001:
        return '*** P=%.4f' % p
    if p < 0.01:
        return '** P=%.3f' % p
    if p < 0.05:
        return '* P=%.3f' % p
    if p >= 0.05:
        return 'ns P=%.2f' % p