import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.save_and_load_data import to_fr
from utils.RRRGD_main_CV import stratify_data_multi_attempts
from utils.RRRGD import RRRGD_model
from utils.utils import compute_R2_main

"""
Function for computing the cross-validated R2 of the null model 
    where the prediction is simply the time-varying mean firing rate of the training data.
"""
def trial_avg_r2(Xy_regression, RRR_df, 
                  nsplit=3, test_size=0.3):
    col_name = f"null_r2"
    if col_name in RRR_df:
        return RRR_df

    res_dict = {"uuids": [], 
                col_name: []}
    def _one_eid(Xy_regression, eid):
        r2s_split = []
        for spliti in range(nsplit):
            Xy_regression[eid] = stratify_data_multi_attempts(Xy_regression[eid], spliti, stratify_by=[[0,2],[0],None], test_size=test_size)
            
            X_test = Xy_regression[eid]['X'][1]
            y_train = Xy_regression[eid]['y'][0]
            y_pred = {eid: np.repeat(np.mean(y_train, 0, keepdims=True), X_test.shape[0], axis=0)}
            
            _, y, y_pred = to_fr(Xy_regression, eid, 
                                Xy_regression[eid]['X'][1],
                                Xy_regression[eid]['y'][1], 
                                y_pred[eid])
            r2s = compute_R2_main(y, y_pred, clip=True)
            r2s_split.append(r2s)
        res_dict['uuids'] += Xy_regression[eid]['setup']['uuids'].tolist()
        res_dict[col_name] += np.mean(r2s_split, 0).tolist()
    
    for eid in tqdm(Xy_regression, desc='iterating over Xy_regression eids'):
        _one_eid(Xy_regression, eid)
    res_df = pd.DataFrame.from_dict(res_dict)

    # merged with uuids
    RRR_df = RRR_df.merge(res_df, left_on=["uuids"], right_on=["uuids"])
    return RRR_df

"""
Function for estimating the timescale of each neuron 
    based on the autocorrelation function of the predicted responses by RRR model.
"""
def load_timescale_neuron(Xy_regression, RRR_df):
    ### tau is defined as the time it takes 
    ### to decay to half of the peak of the autocorrelation function
    res_dict = {"uuids": [], "signal_tau_neuron": []}
    def _one_eid(Xy_regression, eid):
        X = Xy_regression[eid]['Xall']
        beta = np.array(RRR_df.loc[RRR_df.eid==eid, f"RRR_beta"].values.tolist())
        y_signal = RRRGD_model.predict(beta, X, tonp=True)
        taus = compute_neuron_timescale(y_signal)
        
        res_dict["signal_tau_neuron"] += taus.tolist()
        res_dict['uuids'] += Xy_regression[eid]['setup']['uuids'].tolist()
    for eid in tqdm(Xy_regression, desc='iterating over Xy_regression eids'):
        _one_eid(Xy_regression, eid)
        
    res_df = pd.DataFrame.from_dict(res_dict)
    RRR_df = RRR_df.merge(res_df, left_on=["uuids"], right_on=["uuids"])
    return RRR_df

# y: [K, T, N]
# ret: [N] or float if N = 1
def compute_neuron_timescale(y, dt=0.01):
    ## first compute the autocorrelation function
    ### then compute the tau
    K,T,N = y.shape
    taus = np.zeros(N)
    _ts = np.arange(0,T*dt,dt)
    dt2 = 1e-3
    _ts2 = np.arange(0,T*dt,dt2)
    corrs = np.zeros((N, len(_ts2)))
    for ni in range(N):
        corr = np.mean([auto_corr(y[ki,:,ni]) for ki in range(K)], 0)
        corr2 = np.interp(_ts2, _ts, corr)
        _tbin = np.where(corr2 <= corr2[0]/2)[0]
        _tbin = _tbin[0] if len(_tbin) > 0 else np.nan # the first value or an invalid value
        taus[ni] = _tbin * dt2
        corrs[ni] = corr2
    return taus

# * x has to be a 1d array
def auto_corr(x):
    return np.correlate(x,x,mode='full')[len(x)-1:]
