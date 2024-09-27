import numpy as np
import torch
import torch.nn as nn
from utils import np2tensor, np2param, tensor2np, compute_R2_main


# K: number of trials
# T: number of timesteps
# N: number of neurons 
# ncoef: number of coefficients
class RRRGD_model():
    def __init__(self, train_data, ncomp, l2=0.):
        self.n_comp = ncomp
        self.l2=l2
        self.eids = list(train_data.keys())

        np.random.seed(0)
        self.model = {}
        for eid in train_data:
            _X = train_data[eid]['X'][0] # (K,T,ncoef+1), the last coef is the bias term
            _y = train_data[eid]['y'][0] # (K,T,N)
            _, T, ncoef = _X.shape
            ncoef -= 1 # -1 is for the concatenated 1s in X
            _, T, N = _y.shape
            U = np.random.normal(size=(N, ncoef, ncomp))/np.sqrt(T*ncomp) 
            V = np.random.normal(size=(ncomp, T))/np.sqrt(T*ncomp)
            b = np.expand_dims(_y.mean(0).T, 1)
            b = np.ascontiguousarray(b)
            self.model[eid+"_U"]=np2param(U)
            self.model[eid+"_b"]=np2param(b)
        self.model['V'] = np2param(V) # V shared across sessions
        self.model = nn.ParameterDict(self.model)
        # U: model[eid+"_U"], (N, ncoef, ncomp)
        # V: model['V'], (ncomp, T)
        # b: model[eid+"_b"], (N, 1, T)

    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
    def to(self, device):
        self.model.to(device)
    def state_dict(self):
        checkpoint = {"model": {k: v.cpu() for k, v in self.model.state_dict().items()},
                      "eids": self.eids,
                      "l2": self.l2,
                      "n_comp": self.n_comp,}
        return checkpoint
    def load_state_dict(self, f):
        self.model.load_state_dict(f)

    def compute_beta(self, eid, withbias=True):
        U = self.model[eid + "_U"]
        V = self.model['V']
        b = self.model[eid + "_b"]

        beta = U @ V
        if withbias:
            beta = torch.cat((beta, b), 1) # (N, ncoef+1, T)
        
        return beta

    """
    - data {eid: nparray}
    - output tensor
    """
    def predict_y(self, data, eid, k):
        beta = self.compute_beta(eid)
        X = np2tensor(data[eid]['X'][k]).to(beta.device) # (K, T, ncoef+1)
        y = np2tensor(data[eid]['y'][k]).to(beta.device) # (K, T, N)
        ypred = torch.einsum("ktc,nct->ktn", X, beta)
        return X, y, ypred
    
    """
    - data {eid: nparray}
    - output tensor
    """
    def predict_y_fr(self, data, eid, k):
        X, y, ypred = self.predict_y(data, eid, k)
        mean_y = np2tensor(data[eid]['setup']['mean_y_TN']).to(y.device)
        std_y = np2tensor(data[eid]['setup']['std_y_TN']).to(y.device)
        y = y * std_y + mean_y
        ypred = ypred * std_y + mean_y
        return X, y, ypred
    
    """
    - data {eid: nparray}
    - output {eid: tensor}
    """
    def compute_MSE_RRRGD(self, data, k):
        mses_all = {}
        for eid in data:
            _, y, ypred = self.predict_y(data, eid, k)
            mses_all[eid] = torch.sum((ypred - y) ** 2, axis=(0, 1))
        return mses_all

    """
    - output {eid: float}
    """
    def regression_loss(self):
        return {eid: self.l2*torch.sum(self.compute_beta(eid, withbias=False)**2) for eid in self.eids}

    """
    - data {eid: nparray}
    - output {eid: nparray}
    """
    def compute_R2_RRRGD(self, data, k):
        self.eval()
        r2s_all = {}
        for eid in data:
            _, y, ypred = self.predict_y_fr(data, eid, k)
            r2s_val = compute_R2_main(tensor2np(y), 
                                    tensor2np(ypred), 
                                    clip=True)
            r2s_all[eid] = r2s_val
        return r2s_all

"""
train the 
    model: RRRGD_model
given the
    train_data: {eid: {X: (train_X, val_X), 
                       y: (train_y, val_y),
                       setup: dict(mean_y_TN, std_y_TN, mean_X_Tv, std_X_Tv)}}
        where eid is the identity of session
        and train_X, val_X is of shape (#trials, #timesteps, #coefs+1)
        and train_y, val_y is of shape (#trials, #timesteps, #neurons)
        mean_y_TN and std_y_TN are the mean and std of y, of shape (T, N)
        mean_X_Tv and std_X_Tv are the mean and std of X, of shape (T, ncoef)
- model saved in model_fname
"""
def train_model(model, train_data, optimizer, model_fname, save=True):
    def closure():
        optimizer.zero_grad()
        model.train()
        total_loss = 0.0;
        train_mses_all = model.compute_MSE_RRRGD(train_data, 0)
        reg_losses_all = model.regression_loss()
        for eid in train_mses_all:
            total_loss += train_mses_all[eid].sum()
            total_loss += reg_losses_all[eid]
        total_loss.backward()
        return total_loss

    optimizer.step(closure)

    model.eval()
    mses_val = model.compute_MSE_RRRGD(train_data, 1)
    mses_val = {eid: tensor2np(mses_val[eid]) for eid in mses_val}
    mse_val_mean = np.sum(np.concatenate([mses_val[eid] for eid in mses_val]))
    r2s_val = model.compute_R2_RRRGD(train_data, 1)
    r2_val_mean = np.mean(np.concatenate([r2s_val[k] for k in r2s_val]))

    if save:
        # Save the best model parameters
        checkpoint = {"RRRGD_model": model.state_dict(),
                      "optimizer": optimizer.state_dict()}
        torch.save(checkpoint, model_fname)
    
    return model, {"mses_val":mses_val, "mse_val_mean":mse_val_mean, "r2s_val": r2s_val, "r2_val_mean": r2_val_mean}
