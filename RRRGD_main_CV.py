from utils import get_device, log_kv
from RRRGD import RRRGD_model, RRRGD_model_Vdep
from RRRsteinmetzGD import RRRsteinmetzGD_model
from torch import optim
import torch
import numpy as np
import pickle, pdb
from sklearn.model_selection import train_test_split

from utils import tensor2np

 
"""
train the 
    model: RRRGD_model or its inheritence
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
        train_mses_all = model.compute_loss(train_data, 0)
        reg_losses_all = model.regression_loss()
        for eid in train_mses_all:
            total_loss += train_mses_all[eid].sum()
            total_loss += reg_losses_all[eid]
        total_loss.backward()
        return total_loss

    optimizer.step(closure)

    model.eval()
    mses_val = model.compute_loss(train_data, 1)
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


def train_model_main(train_data, RRR_model, model_fname,
                        lr, max_iter, tolerance_grad, tolerance_change, history_size, line_search_fn,
                        save):
    
    device = get_device()
    RRR_model.to(device)
    print(f"training on device: {device}")

    optimizer = optim.LBFGS(RRR_model.model.parameters(),
                            lr=lr,
                            max_iter=max_iter,
                            tolerance_grad=tolerance_grad,
                            history_size=history_size,
                            line_search_fn=line_search_fn,
                            tolerance_change=tolerance_change)
    RRR_model, eval_val = train_model(RRR_model, train_data, optimizer,
                             model_fname=model_fname, save=save)
    return RRR_model, eval_val


"""
split the session data to training and testing set
data is inputed as 
  - data['Xall']: (#trial, #timestep, #covariate), nparray 
  - data['yall']: (#trial, #timestep, #neuron), nparray 
return:
  - data['X']: (X_for_training, X_for_validation), both are nparray of the shape (#trial, #timestep, #covariate)
  - data['y']: (y_for_training, y_for_validation), both are nparray of the shape (#trial, #timestep, #neuron)
"""
def stratify_data(data, spliti, by=None, test_size=0.3):
    if by is None:
        X_train, X_test, y_train, y_test = train_test_split(data["Xall"], data["yall"], 
                                                            test_size=test_size, random_state=42+spliti,)
    
    else:
        X_train, X_test, y_train, y_test = train_test_split(data["Xall"], data["yall"], 
                                                            test_size=test_size, random_state=42+spliti,
                                                            stratify=data["Xall"][:,0,by]) 
    data['X'] = (X_train, X_test)
    data['y'] = (y_train, y_test)
    return data

def stratify_data_multi_attempts(data, spliti, stratify_by=[None], test_size=0.3):
    if stratify_by[-1] is not None: stratify_by.append(None)
    for by in stratify_by:
        try:
            data = stratify_data(data, spliti, by=by, test_size=test_size)
            # if succeed, break the loop
            break
        except:
            # it may fail due to not enough trials for every condition
            # if so, try next stratify_by
            pass
    return data

"""
---- inputs ---
train_data: {eid: {
                    "Xall": (K,T,ncoef+1), # normalized, 1 expanded, 
                    "yall": (K,T,N), # normalized
                    "setup": dict(mean_y_TN, std_y_TN, mean_X_Tv, std_X_Tv)
                    }}
    where eid is the identity of session
    and Xall is of shape (#trials, #timesteps, #coefs+1)
    and yall is of shape (#trials, #timesteps, #neurons)
    mean_y_TN and std_y_TN are the mean and std of y, of shape (T, N)
    mean_X_Tv and std_X_Tv are the mean and std of X, of shape (T, ncoef)
n_comp_list: list of numbers of components/ numbers of temporal basis functions/ ranks
l2_list: list of l2 weights
model_fname: file name for which the model will be saved -> f"{model_fname}.pt"
---- return ---
models_split: list of trained models for each split of data
r2s_cv_best: {eid: (N)} mean validated r2 as the evaluation of performance for each session and neuron
model: trained model using the whole dataset
"""
def train_model_hyper_selection(train_data, n_comp_list, l2_list, model_fname,
                                Vdep=False, RRRsteinmetz=False,
                                lr=1., max_iter=500, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn=None,
                                nsplit=3, test_size=0.3, stratify_by=[None],
                                ):
    # params for LBFGS optimization algorithm
    # the default ones are generally good
    train_params = dict(lr=lr, max_iter=max_iter, 
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size, line_search_fn=line_search_fn)
    
    def _get_model(ncomp, l2):
        if Vdep:
            RRR_model = RRRGD_model_Vdep(train_data, ncomp, l2=l2)
        elif RRRsteinmetz:
            RRR_model = RRRsteinmetzGD_model(train_data, ncomp, l2=l2)
        else:
            RRR_model = RRRGD_model(train_data, ncomp, l2=l2)
        return RRR_model

    #### select the optimal number of components
    ####                    and l2 weight
    #### based on the k-fold cross-validated mean-squared-error (mse)
    best_mse = float('inf'); best_n_comp = None; best_l2 = None
    if len(l2_list) == 1 and len(n_comp_list) == 1:
        ## If we only have one candidate set of hyperparameters
        ## no need for selection
        best_l2 = l2_list[0]
        best_n_comp = n_comp_list[0]
    else:
        for l2 in l2_list:
            for n_comp in n_comp_list:
                ## Cross-Validation:
                mse_val_splits = []; 
                for spliti in range(nsplit):
                    ## split data to training and validation set for each session separately
                    for eid in train_data:
                        train_data[eid] = stratify_data_multi_attempts(train_data[eid], spliti, stratify_by=stratify_by, test_size=test_size)
                    
                    RRR_model = _get_model(n_comp, l2)
                    _, eval_val = train_model_main(train_data, RRR_model, 
                                                    model_fname=None, save=False, 
                                                    **train_params)
                    mse_val_splits.append(eval_val['mse_val_mean'])
                
                ## rewrite the best set of hyperparameters if the mean validated mse is lower
                mse_now = np.mean(mse_val_splits)
                if mse_now < best_mse:
                    best_mse = mse_now
                    best_n_comp = n_comp
                    best_l2 = l2
    log_kv(best_n_comp=best_n_comp, best_l2=best_l2, best_mse=best_mse)

    ## retrain the RRR model with the selected best set of hyperparameters
    ## get and save the mean validated r2 as the evaluation of performance (fname=f"{model_fname}_R2CV.pk")
    ## and save the k-fold model (fname=f"{model_fname}_split{spliti}.pt")
    r2s_split = []; models_split = []
    for spliti in range(nsplit):
        for eid in train_data:
            train_data[eid] = stratify_data_multi_attempts(train_data[eid], spliti, stratify_by=stratify_by, test_size=test_size)

        model_fname_i = f"{model_fname}_split{spliti}.pt" # hard-coded
        RRR_model =  _get_model(best_n_comp, best_l2)
        model_i, eval_val = train_model_main(train_data, RRR_model, 
                                            model_fname=model_fname_i,  
                                            save=True, **train_params)
        r2s_split.append(eval_val['r2s_val'])
        models_split.append(model_i)
    r2s_cv_best = {eid: np.mean([r2s[eid] for r2s in r2s_split], 0) for eid in r2s_split[0]}
    log_kv(r2s_cv_best = np.mean(np.concatenate([r2s_cv_best[eid] for eid in r2s_cv_best])))
    with open(f"{model_fname}_R2test.pk", "wb") as file:  # hard-coded
        pickle.dump(r2s_cv_best, file)

    ## retrain the RRR model *on the whole dataset* with the selected best set of hyperparameters
    ## get and save the model (fname=f"{model_fname}.pt")
    for eid in train_data:
        # train (and validate) on the whole dataset
        # the evaluation result will not be counted
        train_data[eid]["X"] = (train_data[eid]["Xall"], train_data[eid]["Xall"])
        train_data[eid]["y"] = (train_data[eid]["yall"], train_data[eid]["yall"])
    RRR_model =  _get_model(best_n_comp, best_l2)
    model, _ = train_model_main(train_data, RRR_model,
                                model_fname=f"{model_fname}.pt",
                                save=True, **train_params)

    return models_split, r2s_cv_best, model

