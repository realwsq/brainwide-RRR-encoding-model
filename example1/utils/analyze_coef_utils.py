from example1.utils.analyze_perf_utils import ensemble_prediction, to_fr
from example1.utils.save_and_load_data import iterate_over_eids
from utils import load_or_save_df

import numpy as np
import pandas as pd
import os, pdb

# * x has to be a 1d array
def auto_corr(x, delay_max=None):
    if (delay_max is not None) and (delay_max < len(x)):
        return np.correlate(x,x[:-delay_max],mode='full')[len(x)-delay_max-1:len(x)-1]
    else:
        return np.correlate(x,x,mode='full')[len(x)-1:]

# y: [K, T, N]
# ret: [N] or float if N = 1
def compute_neuron_timescale(y, dt=0.01, delay_max=None):
    ## first compute the autocorrelation function
    ### then compute the tau
    K,T,N = y.shape
    taus = np.zeros(N)
    if delay_max is None: delay_max = T
    _ts = np.arange(0,delay_max*dt,dt)
    dt2 = 1e-3
    _ts2 = np.arange(0,delay_max*dt,dt2)
    corrs = np.zeros((N, len(_ts2)))
    for ni in range(N):
        corr = np.mean([auto_corr(y[ki,:,ni], delay_max) for ki in range(K)], 0)
        corr2 = np.interp(_ts2, _ts, corr)
        _tbin = np.where(corr2 <= corr2[0]/2)[0]
        _tbin = _tbin[0] if len(_tbin) > 0 else np.nan # the first value or an invalid value
        taus[ni] = _tbin * dt2
        corrs[ni] = corr2
    return taus, corrs

def load_timescale_neuron(Xy_regression, RRR_res_df, 
                          mname='RRRglobal', local_folder=None, dt=0.01, delay_max=None, ytofr=False,
                          **pred_kwargs):
    ### tau is defined as the time it takes 
    ### to decay to half of the peak of the autocorrelation function
    postfix = f"tau_neuron {mname}_{delay_max}" if (ytofr == False) else f"taufr_neuron {mname}_{delay_max}" 
    _list = ['signal', 'noise', 'all']
    def _main():
        res_dict = {"uuids": [],}
        for s in _list:
            res_dict[f"{s}_{postfix}"] = []
        def _one_eid(Xy_regression_eids, eid):
            y_pred = ensemble_prediction(Xy_regression_eids, RRR_res_df, mname, eid,
                                        local_folder=local_folder, cv=False, 
                                        **pred_kwargs)
            _y_signal = y_pred[eid]
            _y_all = Xy_regression_eids[eid]['yall']
            if ytofr:
                _, _y_all, _y_signal = to_fr(Xy_regression_eids, eid, 
                                          Xy_regression_eids[eid]['Xall'], _y_all, _y_signal)
            _y_noise = _y_all - _y_signal
            for s in _list:
                if s == "signal":
                    taus, _ = compute_neuron_timescale(_y_signal, dt=dt, delay_max=delay_max)
                elif s == 'noise':
                    taus, _ = compute_neuron_timescale(_y_noise, dt=dt, delay_max=delay_max)
                elif s == 'all':
                    taus, _ = compute_neuron_timescale(_y_all, dt=dt, delay_max=delay_max)
                else:
                    assert False, "invalid s"
                res_dict[f"{s}_{postfix}"] += taus.tolist()
            res_dict['uuids'] += Xy_regression_eids[eid]['setup']['uuids'].tolist()
        iterate_over_eids(Xy_regression, _one_eid)
        temp_res_df = pd.DataFrame.from_dict(res_dict)
        return temp_res_df 
    if local_folder is not None: # try to load/ save results to local folder
        _fname = os.path.join(local_folder, f"{postfix}.csv")
        temp_res_df = load_or_save_df(_fname, _main)
    else:
        temp_res_df = _main()
    RRR_res_df = RRR_res_df.merge(temp_res_df, left_on=["uuids"], right_on=["uuids"])
    return RRR_res_df

