from utils import get_device
from RRRGD import train_model, RRRGD_model, RRRGD_model_Vdep
from torch import optim
import numpy as np
import pickle, pdb
from sklearn.model_selection import train_test_split


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
def stratify_data_random(data, spliti, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(data["Xall"], data["yall"], 
                                                    test_size=test_size, random_state=42+spliti,)
    data['X'] = (X_train, X_test)
    data['y'] = (y_train, y_test)
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
                                Vdep=False,
                                lr=1., max_iter=500, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn=None,
                                nsplit=3, test_size=0.3,
                                ):
    # params for LBFGS optimization algorithm
    # the default ones are generally good
    train_params = dict(lr=lr, max_iter=max_iter, 
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size, line_search_fn=line_search_fn)
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
                        train_data[eid] = stratify_data_random(train_data[eid], spliti, test_size=test_size)

                    if Vdep:
                        RRR_model = RRRGD_model_Vdep(train_data, n_comp, l2=l2)
                    else:
                        RRR_model = RRRGD_model(train_data, n_comp, l2=l2)
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

    ## retrain the RRR model with the selected best set of hyperparameters
    ## get and save the mean validated r2 as the evaluation of performance (fname=f"{model_fname}_R2CV.pk")
    ## and save the k-fold model (fname=f"{model_fname}_split{spliti}.pt")
    r2s_split = []; models_split = []
    for spliti in range(nsplit):
        for eid in train_data:
            train_data[eid] = stratify_data_random(train_data[eid], spliti, test_size=test_size)

        model_fname_i = f"{model_fname}_split{spliti}.pt" # hard-coded
        if Vdep:
            RRR_model = RRRGD_model_Vdep(train_data, best_n_comp, l2=best_l2)
        else:
            RRR_model = RRRGD_model(train_data, best_n_comp, l2=best_l2)
        model_i, eval_val = train_model_main(train_data, RRR_model, 
                                                model_fname=model_fname_i,  
                                                save=True, **train_params)
        r2s_split.append(eval_val['r2s_val'])
        models_split.append(model_i)
    r2s_cv_best = {eid: np.mean([r2s[eid] for r2s in r2s_split], 0) for eid in r2s_split[0]}
    with open(f"{model_fname}_R2test.pk", "wb") as file:  # hard-coded
        pickle.dump(r2s_cv_best, file)

    ## retrain the RRR model *on the whole dataset* with the selected best set of hyperparameters
    ## get and save the model (fname=f"{model_fname}.pt")
    for eid in train_data:
        # train (and validate) on the whole dataset
        # the evaluation result will not be counted
        train_data[eid]["X"] = (train_data[eid]["Xall"], train_data[eid]["Xall"])
        train_data[eid]["y"] = (train_data[eid]["yall"], train_data[eid]["yall"])
    model, _ = train_model_main(train_data, best_l2, best_n_comp,
                                model_fname=f"{model_fname}.pt",
                                save=True, **train_params)

    return models_split, r2s_cv_best, model

