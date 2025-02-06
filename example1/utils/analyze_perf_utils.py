from utils import load_or_save_dict, load_or_save_df, remove_space, compute_R2_main
from example1.utils.save_and_load_data import iterate_over_eids
from RRRGD import RRRGD_model
from RRRGD_main_CV import stratify_data_multi_attempts

import os, pdb
import numpy as np
import pandas as pd


"""
* target one specific eid
"""
def ensemble_prediction_XRR(RRR_res_df, Xy_regression, target_eid, mname="RRRglobal",
                            test_only=True, local_folder=None, spliti=-1,
                            **others):
    def _main(RRR_res_df):
        y_pred = {}
        if test_only:
            X = Xy_regression[target_eid]['X'][1]
            beta = np.array(RRR_res_df.loc[RRR_res_df.eid==target_eid, f"{mname}{spliti}_beta"].values.tolist())
        else:
            X = Xy_regression[target_eid]['Xall']
            beta = np.array(RRR_res_df.loc[RRR_res_df.eid==target_eid, f"{mname}_beta"].values.tolist())
        y_pred[target_eid] = RRRGD_model.predict(beta, X, tonp=True)
        return y_pred
    if local_folder is not None:
        _fname = os.path.join(local_folder, f"{mname}_pred_{target_eid}_{spliti}_{test_only}.pk")
        y_pred = load_or_save_dict(_fname, _main, RRR_res_df=RRR_res_df)
    else:
        y_pred = _main(RRR_res_df)
    return y_pred


"""
- X, y should be nparray with
    - X: [K,T,ncoef]
    - y: [K,T,N] or [K,T]
- axis and value should be list/ nparray
- return: nparray [T, N] or [T]
"""
def compute_PSTH(X, y, axis, value):
    trials = np.all(np.isclose(X[:, 0, axis], value), axis=-1)
    if np.sum(trials) == 0:
        print(f"Be careful !! No such condition found {axis}, {value}")
        trials = np.arange(y.shape[0])
    return y[trials].mean(0)


def pred_wPSTH(X_train, y_train, X_test, idxs_psth=None):
    y_pred = np.zeros((X_test.shape[0], y_train.shape[1], y_train.shape[2]))
    for k in range(X_test.shape[0]):
        y_pred[k] = compute_PSTH(X_train, y_train, axis=idxs_psth, value=X_test[k,0,idxs_psth])
    return y_pred


def ensemble_prediction_taskpsth(Xy_regression, idxs_psth, target_eid,
                                 test_only=True, local_folder=None, spliti=-1,
                                 **others):
    assert idxs_psth is not None, "idxs_psth cannot be None, must be a list"
    def _main():
        y_pred = {}
        if test_only:
            X_train = Xy_regression[target_eid]['X'][0]
            X_test = Xy_regression[target_eid]['X'][1]
            y_train = Xy_regression[target_eid]['y'][0]
        else:
            X_train = X_test = Xy_regression[target_eid]['Xall']
            y_train = Xy_regression[target_eid]['yall']
        y_pred[target_eid] = pred_wPSTH(X_train, y_train, X_test, idxs_psth=idxs_psth) 
        return y_pred
    if local_folder is not None: # try to load/ save results to local folder
       _fname = os.path.join(local_folder, f"taskpsth_pred_{target_eid}_{spliti}_{test_only}.pk")
       y_pred = load_or_save_dict(_fname, _main)
    else:
        y_pred = _main()
    return y_pred


def ensemble_prediction_meanact(Xy_regression, target_eid,
                                 test_only=True, local_folder=None, spliti=-1,
                                 **others):
    def _main():
        y_pred = {}
        if test_only:
            X_test = Xy_regression[target_eid]['X'][1]
            y_train = Xy_regression[target_eid]['y'][0]
            y_test = Xy_regression[target_eid]['y'][1]
        else:
            X_test = Xy_regression[target_eid]['Xall']
            y_train = Xy_regression[target_eid]['yall']
            y_test = Xy_regression[target_eid]['yall']
        y_pred[target_eid] = np.repeat(np.mean(y_train, 0, keepdims=True), X_test.shape[0], axis=0) # previously used
        # y_pred[target_eid] = np.zeros_like(y_test)
        return y_pred
    if local_folder is not None: # try to load/ save results to local folder
       _id = remove_space(f"{Xy_regression[target_eid]['yall'].shape}")
       _fname = os.path.join(local_folder, f"meanact_pred_{target_eid}_{_id}_{spliti}_{test_only}.pk")
       y_pred = load_or_save_dict(_fname, _main)
    else:
        y_pred = _main()
    return y_pred


def ensemble_prediction(Xy_regression, RRR_res_df, mname, target_eid, 
                        cv=True, spliti=0, test_size=0.3, 
                        local_folder=None, **pred_kwargs):
    # limited space, only save results for RRRglobal
    if (local_folder is not None) and (mname in ["RRRglobal"]):
        pass
    else:
        local_folder = None

    if cv:
        Xy_regression[target_eid] = stratify_data_multi_attempts(Xy_regression[target_eid], spliti, stratify_by=[[0,2],[0],None], test_size=test_size)
        # Xy_regression[target_eid] = stratify_data_random(Xy_regression[target_eid], spliti, test_size=test_size)

    if mname in ["RRRglobal"]:
        y_pred = ensemble_prediction_XRR(RRR_res_df, Xy_regression, mname=mname, target_eid=target_eid,
                                         local_folder=local_folder, test_only=cv, spliti=spliti)
    elif mname == "taskpsth":
        y_pred = ensemble_prediction_taskpsth(Xy_regression, pred_kwargs['idxs_psth'], target_eid, 
                                              local_folder=local_folder, test_only=cv, spliti=spliti)
    elif mname.startswith("meanact"):
        y_pred = ensemble_prediction_meanact(Xy_regression, target_eid,
                                             local_folder=local_folder, test_only=cv, spliti=spliti)
    else:
        assert False, "invalid mname"

    return y_pred

def to_fr(Xy_eids, eid, X, y, y_pred):
    mean_y_TN = Xy_eids[eid]['setup']['mean_y_TN']
    std_y_TN = Xy_eids[eid]['setup']['std_y_TN']
    y_pred = y_pred*std_y_TN + mean_y_TN
    y = y*std_y_TN + mean_y_TN
    X[:,:,:-1] = X[:,:,:-1]*Xy_eids[eid]['setup']['std_X_Tv'] + Xy_eids[eid]['setup']['mean_X_Tv']
    return X, y, y_pred

   
    
def load_xpsth_r2(Xy_regression, res_df, mname="taskbehpsth10", local_folder=None, 
                  nsplit=3, test_size=0.3,
                  **pred_kwargs):
    col_name = f"{mname}_r2"
    if col_name in res_df:
        return res_df

    def _main():
        res_dict = {"uuids": [], 
                    col_name: []}
        def _one_eid(Xy_regression_eids, eid):
            r2s_split = []
            for spliti in range(nsplit):
                y_pred = ensemble_prediction(Xy_regression_eids, None, mname, eid,
                                            local_folder=local_folder, test_only=True, spliti=spliti, test_size=test_size,
                                            **pred_kwargs)
                _, y, y_pred = to_fr(Xy_regression_eids, eid, 
                                    Xy_regression_eids[eid]['X'][1],
                                    Xy_regression_eids[eid]['y'][1], 
                                    y_pred[eid])
                r2s = compute_R2_main(y, y_pred, clip=True)
                r2s_split.append(r2s)
            res_dict['uuids'] += Xy_regression_eids[eid]['setup']['uuids'].tolist()
            res_dict[col_name] += np.mean(r2s_split, 0).tolist()
        iterate_over_eids(Xy_regression, _one_eid)
        temp_res_df = pd.DataFrame.from_dict(res_dict)
        return temp_res_df
    if local_folder is not None: # try to load/ save results to local folder
        _fname = os.path.join(local_folder, f"{col_name}.csv")
        temp_res_df = load_or_save_df(_fname, _main)
    else:
        temp_res_df = _main()

    # merged with uuids
    res_df = res_df.merge(temp_res_df, left_on=["uuids"], right_on=["uuids"])
    return res_df

