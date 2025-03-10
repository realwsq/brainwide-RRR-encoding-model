import numpy as np
import os, pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import np2tensor, np2param, tensor2np, compute_R2_main


# K: number of trials
# T: number of timesteps
# N: number of neurons 
# ncoef: number of coefficients/ variables
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

    """
    * input has to be tensor
    """
    @classmethod
    def compute_beta_m(cls, U, V, b, withbias=True, tonp=False):
        if tonp == True:
            U = np2tensor(U)
            V = np2tensor(V)
        beta = U @ V
        if withbias:
            if tonp == True:
                b = np2tensor(b)
            beta = torch.cat((beta, b), 1) # (N, ncoef+1, T)
        else:
            pass 
        if tonp == True:
            beta = tensor2np(beta)
        
        return beta

    def compute_beta(self, eid, withbias=True):
        return RRRGD_model.compute_beta_m(self.model[eid + "_U"],
                                    self.model['V'],
                                    self.model[eid + "_b"],
                                    withbias=withbias)


    ### main function of performing prediction from X to y_pred
    @classmethod
    def predict(cls, beta, X, tonp=False):
        """
        :beta: (N, ncoef, T)
        :X: (K, T, ncoef)
        """
        if tonp == True:
            X = np2tensor(X)
            beta = np2tensor(beta)
        y_pred = torch.einsum("ktc,nct->ktn", X, beta)
        if tonp == True:
            y_pred = tensor2np(y_pred)
        return y_pred
    
    """
    - data {eid: nparray}
    - output tensor
    """
    def predict_y(self, data, eid, k):
        beta = self.compute_beta(eid)
        if k is not None:
            X = np2tensor(data[eid]['X'][k]).to(beta.device) # (K, T, ncoef+1)
            y = np2tensor(data[eid]['y'][k]).to(beta.device) # (K, T, N)
        else:
            X = np2tensor(data[eid]['Xall']).to(beta.device) # (K, T, ncoef+1)
            y = np2tensor(data[eid]['yall']).to(beta.device) # (K, T, N)
        ypred = RRRGD_model.predict(beta, X)
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

    def compute_loss(self, data, k):
        return self.compute_MSE_RRRGD(data, k)

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

    
    @classmethod
    def params2RRRres(cls, fname, has_area=False):
        res = torch.load(f"{fname}.pt", map_location=torch.device('cpu'))
        best_l2 = res["RRRGD_model"]["l2"]
        best_ncomp = res["RRRGD_model"]["n_comp"]
        res = res["RRRGD_model"]["model"]
        if os.path.isfile(fname + "_R2test.pk"):
            with open(fname + "_R2test.pk", "rb") as file:  # hard-coded
                r2s = pickle.load(file)
        else:
            r2s = None
        res_dict = {}
        for k in tqdm(res.keys(), desc="params2RRRres"):
            if (k == "V"):
                pass
            else:
                V = tensor2np(res['V'])
                eid, k_ = k.split('_')
                if (eid in res_dict):
                    pass
                U = tensor2np(res[f"{eid}_U"])
                b = tensor2np(res[f"{eid}_b"])
                beta = RRRGD_model.compute_beta_m(U, V, b, tonp=True)  # (N, ncoef+1, T)
                res_dict[eid] = {"model": {"beta": beta,
                                            "U": U,
                                            "V": V,
                                            "b": b,
                                            "best_ncomp": best_ncomp,
                                            "best_l2": best_l2},
                                    "r2": r2s[eid] if r2s is not None else np.zeros(len(beta))}

        return res_dict

