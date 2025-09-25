from utils import get_device, log_kv, np2tensor
from RRRGD import RRRGD_model
from torch import optim
import torch
from torch import nn
import numpy as np
import pickle, os
from sklearn.model_selection import train_test_split

from utils import tensor2np

 
"""
train the 
    model: RRRGD_model or its inheritence
given the
    data_all_tv: {eid: {X: (train_X, val_X), 
                       y: (train_y, val_y),
                       setup: dict(mean_y_TN, std_y_TN, mean_X_Tv, std_X_Tv)}}
        where eid is the identity of session
        and train_X, val_X is of shape (#trials, #timesteps, #coefs+1)
        and train_y, val_y is of shape (#trials, #timesteps, #neurons)
        mean_y_TN and std_y_TN are the mean and std of y, of shape (T, N)
        mean_X_Tv and std_X_Tv are the mean and std of X, of shape (T, ncoef)
- model saved in model_fname
"""
def train_model(model, data_all_tv, optimizer):
    def closure():
        optimizer.zero_grad()
        model.train()
        total_loss = 0.0;
        train_mses_all = model.compute_MSE_loss(data_all_tv, 0)
        reg_losses_all = model.regression_loss()
        for eid in train_mses_all:
            total_loss += train_mses_all[eid].sum()
            total_loss += reg_losses_all[eid]
        total_loss.backward()
        closure_calls[0] += 1
        print(f"step: {closure_calls[0]} total_loss: {total_loss.item()}")
        return total_loss

    closure_calls = [0]  
    total_loss = optimizer.step(closure)

    return model, total_loss

def eval_model(model, data_all_tv):
    model.eval()
    model.sample_sessions(None)
    model.to("cpu") # move to cpu for evaluation
    mses_val = model.compute_MSE_loss(data_all_tv, 1)
    mses_val = {eid: tensor2np(mses_val[eid]) for eid in mses_val}
    mse_val_mean = np.sum(np.concatenate([mses_val[eid] for eid in mses_val]))
    r2s_val = model.compute_R2_RRRGD(data_all_tv, 1)
    r2_val_mean = np.mean(np.concatenate([r2s_val[k] for k in r2s_val]))
    print(f"mse_val_mean: {mse_val_mean}, r2_val_mean: {r2_val_mean}")
    model.to(get_device())  # move back to "gpu"
    return  {"mses_val":mses_val, "mse_val_mean":mse_val_mean, "r2s_val": r2s_val, "r2_val_mean": r2_val_mean}


def save_model(model, optimizer, model_fname):
    # Save the best model parameters
    checkpoint = {"RRRGD_model": model.state_dict(),
                  "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, model_fname)
    print(f"Model saved to {model_fname}")

def train_model_main(data_all_tv, RRR_model, model_fname,
                    lr, max_iter, tolerance_grad, tolerance_change, history_size, line_search_fn,
                    save):
    
    device = get_device()
    RRR_model.to(device)

    optimizer = optim.LBFGS(RRR_model.model.parameters(),
                            lr=lr,
                            max_iter=max_iter,
                            tolerance_grad=tolerance_grad,
                            history_size=history_size,
                            line_search_fn=line_search_fn,
                            tolerance_change=tolerance_change)
    RRR_model, _ = train_model(RRR_model, data_all_tv, optimizer)
    eval_val = eval_model(RRR_model, data_all_tv)

    if save:
        # Save the best model parameters
        save_model(RRR_model, optimizer, model_fname)
    
    return RRR_model, eval_val


def train_model_main_sampled(data_all_tv, RRR_model, model_fname, max_epochs, n_sess, patience,
                        lr, max_iter, tolerance_grad, tolerance_change, history_size, line_search_fn,
                        save):
    device = get_device()
    RRR_model.to(device)
    print(f"training on device: {device}")

    best_val = 0.; best_val_epoch = -1
    model_fname_ms_temp = model_fname[:-3]+"sampled_ms_temp.pt" # hard-coded
    for _ in range(max_epochs):
        print(f"epoch: {_+1}/{max_epochs}")
        RRR_model.sample_sessions(n_sess)

        optimizer = optim.LBFGS(RRR_model.model.parameters(),
                                lr=lr,
                                max_iter=max_iter,
                                tolerance_grad=tolerance_grad,
                                history_size=history_size,
                                line_search_fn=line_search_fn,
                                tolerance_change=tolerance_change)
        RRR_model, tot_loss = train_model(RRR_model, data_all_tv, optimizer)
        eval_val = eval_model(RRR_model, data_all_tv)

        if eval_val['r2_val_mean'] > best_val:
            best_val = eval_val['r2_val_mean']
            best_val_epoch = _
            print(f"best_val: {best_val} at epoch: {best_val_epoch}")
            
            # Save the best model parameters
            save_model(RRR_model, optimizer, model_fname_ms_temp) # hard-coded

            if _ - best_val_epoch >= patience:
                break
        
    RRR_model.load_state_dict(model_fname_ms_temp)
    eval_val = eval_model(RRR_model, data_all_tv)
    if save:
        # Save the best model parameters
        save_model(RRR_model, optimizer, model_fname)
        
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

def stratify_data_multi_attempts(data, spliti, stratify_by=[None], test_size=0.3, to_tensor=False):
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
    if to_tensor:
        data["X"] = (np2tensor(data["X"][0]), np2tensor(data["X"][1]))
        data["y"] = (np2tensor(data["y"][0]), np2tensor(data["y"][1]))
    return data

"""
---- inputs ---
data_all: {eid: {
                    "Xall": (K,T,ncoef+1), # normalized across trials, 1 expanded at the end for the learning of the bias, 
                    "yall": (K,T,N), # normalized across trials
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
def train_model_hyper_selection(data_all, n_comp_list, l2_list, model_fname,
                                lr=1., max_iter=500, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn=None,
                                nsplit=3, test_size=0.3, stratify_by=[None],
                                sample_kwargs={"sample": False},
                                **others,
                                ):
    # params for LBFGS optimization algorithm
    # the default ones are generally good
    train_params = dict(lr=lr, max_iter=max_iter, 
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size, line_search_fn=line_search_fn)

    def _train_model(n_comp, l2, model_fname, save=True):
        RRR_model = RRRGD_model(data_all, n_comp=n_comp, l2=l2)
        if sample_kwargs["sample"] == False:
            model_i, eval_val = train_model_main(data_all, RRR_model, 
                                                model_fname=model_fname,  
                                                save=save, 
                                                **train_params)
        else:
            model_i, eval_val = train_model_main_sampled(data_all, RRR_model, 
                                                        model_fname,
                                                        sample_kwargs['max_epochs'], sample_kwargs['n_sess'], sample_kwargs['patience'], 
                                                        save=save, 
                                                        **train_params)
        return model_i, eval_val

    #### select the optimal number of components
    ####                    and l2 weight
    #### based on the k-fold cross-validated mean-squared-error (mse)
    best_r2 = -float('inf'); best_n_comp = None; best_l2 = None
    if len(l2_list) == 1 and len(n_comp_list) == 1:
        ## If we only have one candidate set of hyperparameters
        ## no need for selection
        best_l2 = l2_list[0]
        best_n_comp = n_comp_list[0]
    else:
        for l2 in l2_list:
            for n_comp in n_comp_list:
                ## Cross-Validation:
                mse_val_splits = []; r2_val_splits = []
                for spliti in range(nsplit):
                    ## split data to training and validation set for each session separately
                    for eid in data_all:
                        data_all[eid] = stratify_data_multi_attempts(data_all[eid], spliti, stratify_by=stratify_by, test_size=test_size,
                                                                     to_tensor=True)

                    model_fname_i = f"{model_fname}_l2{l2}_n_comp{n_comp}_split{spliti}.pt" # hard-coded
                    log_kv(model_fname_i=model_fname_i, l2=l2, n_comp=n_comp, spliti=spliti)
                    if os.path.isfile(model_fname_i):
                        model_i = RRRGD_model(data_all, n_comp=n_comp, l2=l2)
                        model_i.load_state_dict(model_fname_i)
                        eval_val = eval_model(model_i, data_all)
                    else:
                        model_i, eval_val = _train_model(n_comp, l2, model_fname_i)
                    # _, eval_val = _train_model(n_comp, l2, model_fname_i, save=False)
                    mse_val_splits.append(eval_val['mse_val_mean'])
                    r2_val_splits.append(eval_val['r2_val_mean'])
                
                ## rewrite the best set of hyperparameters if the mean validated mse is lower
                mse_now = np.mean(mse_val_splits)
                r2_now = np.mean(r2_val_splits)
                log_kv(l2=l2, n_comp=n_comp, mse_now=mse_now, r2_now=r2_now)
                # either select by mse or r2
                # both work
                # r2 is probably better
                if r2_now > best_r2: 
                    best_r2 = r2_now
                    best_n_comp = n_comp
                    best_l2 = l2
    log_kv(best_n_comp=best_n_comp, best_l2=best_l2, best_r2=best_r2)

    # # retrain the RRR model with the selected best set of hyperparameters
    # # get and save the mean validated r2 as the evaluation of performance (fname=f"{model_fname}_R2CV.pk")
    # # and save the k-fold model (fname=f"{model_fname}_split{spliti}.pt")
    r2s_split = []; models_split = []
    for spliti in range(nsplit):
        for eid in data_all:
            data_all[eid] = stratify_data_multi_attempts(data_all[eid], spliti, stratify_by=stratify_by, test_size=test_size,
                                                         to_tensor=True)

        model_fname_i = f"{model_fname}_split{spliti}.pt" # hard-coded
        if os.path.isfile(model_fname_i):
            model_i = RRRGD_model(data_all, n_comp=best_n_comp, l2=best_l2)
            model_i.load_state_dict(model_fname_i)
            eval_val = eval_model(model_i, data_all)
        else:
            model_i, eval_val = _train_model(best_n_comp, best_l2, model_fname_i)
        r2s_split.append(eval_val['r2s_val'])
        models_split.append(model_i)
    r2s_cv_best = {eid: np.mean([r2s[eid] for r2s in r2s_split], 0) for eid in r2s_split[0]}
    log_kv(r2s_cv_best = np.mean(np.concatenate([r2s_cv_best[eid] for eid in r2s_cv_best])))
    with open(f"{model_fname}_R2test.pk", "wb") as file:  # hard-coded
        pickle.dump(r2s_cv_best, file)

    ## retrain the RRR model *on the whole dataset* with the selected best set of hyperparameters
    ## get and save the model (fname=f"{model_fname}.pt")
    for eid in data_all:
        # train (and validate) on the whole dataset
        # the evaluation result will not be counted
        data_all[eid]["X"] = (np2tensor(data_all[eid]["Xall"]), np2tensor(data_all[eid]["Xall"]))
        data_all[eid]["y"] = (np2tensor(data_all[eid]["yall"]), np2tensor(data_all[eid]["yall"]))
    model, _ = _train_model(best_n_comp, best_l2, f"{model_fname}.pt")

    return models_split, r2s_cv_best, model

