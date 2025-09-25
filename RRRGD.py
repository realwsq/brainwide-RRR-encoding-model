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
    def __init__(self, train_data, n_comp, l2=0.):
        self.n_comp = n_comp
        self.l2=l2
        self.eids = list(train_data.keys())
        self.eid_list_train = self.eids

        np.random.seed(0)
        self.model = {}
        for eid in train_data:
            _X = train_data[eid]['X'][0] # (K,T,ncoef+1), the last coef is the bias term
            _y = train_data[eid]['y'][0] # (K,T,N)
            _, T, ncoef = _X.shape
            ncoef -= 1 # -1 is for the concatenated 1s in X
            _, T, N = _y.shape
            U = np.random.normal(size=(N, ncoef, n_comp))/np.sqrt(T*n_comp) 
            V = np.random.normal(size=(n_comp, T))/np.sqrt(T*n_comp)
            b = np.expand_dims(_y.mean(0).T, 1)
            b = np.ascontiguousarray(b)
            self.model[eid+"_U"]=np2param(U)
            self.model[eid+"_b"]=np2param(b)
        self.model['V'] = np2param(V) # V shared across sessions
        self.model = nn.ParameterDict(self.model)
        # U: model[eid+"_U"], (N, ncoef, n_comp)
        # V: model['V'], (n_comp, T)
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
        print(f)
        self.model.load_state_dict(torch.load(f, weights_only=False)['RRRGD_model']['model'])

    
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
        if isinstance(self.model, nn.DataParallel):
            U = self.model.module[eid + "_U"]
            V = self.model.module['V']
            b = self.model.module[eid + "_b"]
        else:
            U = self.model[eid + "_U"]
            V = self.model['V']
            b = self.model[eid + "_b"]
        return RRRGD_model.compute_beta_m(U, V, b,
                                    withbias=withbias)


    ### main function of performing prediction from X to y_pred
    @classmethod
    def predict(cls, beta, X, tonp=False):
        """
        :beta: (N, ncoef, T)
        :X: (K, T, ncoef)
        """
        if type(X) is np.ndarray:
            X = np2tensor(X)
        if type(beta) is np.ndarray:
            beta = np2tensor(beta)
        y_pred = torch.einsum("ktc,nct->ktn", X, beta)
        if tonp == True:
            y_pred = tensor2np(y_pred)
        return y_pred
    
    """
    - data {eid: nparray or tensor}
    - output tensor
    """
    def predict_y(self, data, eid, k):
        beta = self.compute_beta(eid)
        if k is not None:
            X = data[eid]['X'][k] # (K, T, ncoef+1)
            y = data[eid]['y'][k] # (K, T, N)
        else:
            X = data[eid]['Xall'] # (K, T, ncoef+1)
            y = data[eid]['yall']
        if type(X) is np.ndarray:
            X = np2tensor(X) # (K, T, ncoef+1)
            y = np2tensor(y) # (K, T, N)        
        if beta.is_cuda:
            X = X.to(beta.device)
            y = y.to(beta.device)
        ypred = RRRGD_model.predict(beta, X)
        return X, y, ypred
    
    """
    - data {eid: nparray or tensor}
    - output tensor
    """
    def predict_y_fr(self, data, eid, k):
        X, y, ypred = self.predict_y(data, eid, k)
        mean_y = np2tensor(data[eid]['setup']['mean_y_TN'])
        std_y = np2tensor(data[eid]['setup']['std_y_TN'])
        if y.is_cuda:
            mean_y = mean_y.to(y.device)
            std_y = std_y.to(y.device)
        y = y * std_y + mean_y
        ypred = ypred * std_y + mean_y
        return X, y, ypred
    
    """
    - data {eid: nparray}
    - output {eid: tensor}
    """
    def compute_MSE_loss(self, data, k):
        mses_all = {}
        for eid in self.eid_list_train:
            _, y, ypred = self.predict_y(data, eid, k)
            mses_all[eid] = torch.sum((ypred - y) ** 2, axis=(0, 1))
        return mses_all

    """
    - output {eid: float}
    """
    def regression_loss(self):
        return {eid: self.l2*torch.sum(self.compute_beta(eid, withbias=False)**2) for eid in self.eid_list_train}

    def sample_sessions(self, n_sess):
        if (n_sess is not None) and (n_sess <= len(self.eids)):
            self.eid_list_train = np.random.choice(self.eids, n_sess, replace=False)
        else:
            self.eid_list_train = self.eids
        print(len(self.eid_list_train), "sampled eids")

    """
    - data {eid: nparray}
    - output {eid: nparray}
    """
    def compute_R2_RRRGD(self, data, k):
        self.eval()
        r2s_all = {}
        # for eid in data:
        for eidi, eid in enumerate(self.eid_list_train):
            _, y, ypred = self.predict_y_fr(data, eid, k)
            r2s_val = compute_R2_main(tensor2np(y), 
                                    tensor2np(ypred), 
                                    clip=True)
            r2s_all[eid] = r2s_val
        return r2s_all

    @classmethod
    def params2RRRres(cls, fname, has_area=False):
        res = torch.load(f"{fname}.pt", map_location=torch.device('cpu'), weights_only=False)
        best_l2 = res["RRRGD_model"]["l2"]
        best_n_comp = res["RRRGD_model"]["n_comp"]
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
                                            "best_n_comp": best_n_comp,
                                            "best_l2": best_l2},
                                    "r2": r2s[eid] if r2s is not None else np.zeros(len(beta))}

        return res_dict

