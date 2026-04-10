import os
import pandas as pd
from utils.save_and_load_data import read_Xy_4_RRR, load_df_from_Xy_regression_setup
from utils.RRRGD import RRRGD_model
from utils.analyze_perf_utils import trial_avg_r2

"""
load the RRRGD model
"""
def load_RRR_res(RRR_fname):
    columns = ["eid", "ni", "RRR_beta", "RRR_r2"] 
    columns += ["RRR_U", "RRR_V", "RRR_b"]
    res_dict_temp = {c: [] for c in columns}

    if os.path.isfile(f"{RRR_fname}.pt"):
        res_all = RRRGD_model.params2RRRres(RRR_fname) 
    else:
        assert False, f"no file {RRR_fname}"
    
    for eid in sorted(list(res_all.keys())):
        res = res_all[eid]
        N = len(res["r2"])
        res_dict_temp['eid'] += [eid] * N
        res_dict_temp['ni'] += list(range(N))
        res_dict_temp["RRR_r2"] += res["r2"].tolist()
        res_dict_temp["RRR_beta"] += [res["model"]["beta"][ni] for ni in range(N)] # (ncoef, T)
        res_dict_temp["RRR_U"] += [res["model"]["U"][ni] for ni in range(N)] # (ncoef, ncomp)
        res_dict_temp["RRR_V"] += [res["model"]["V"]] * N # (ncomp, T)
        res_dict_temp["RRR_b"] += [res["model"]["b"][ni][0] for ni in range(N)] # (T)
    
    RRR_res_df = pd.DataFrame.from_dict(res_dict_temp)
    return RRR_res_df

res_fname = os.path.join("./trained_RRR_model", "RRR_selectivity.json")

### load the data and the trained model
# load the trained model
RRR_fname = os.path.join("./trained_RRR_model", f"RRRGD") 
RRR_res_df = load_RRR_res(RRR_fname)
# to merge the uuids, acronym, mfr_task from the data to RRR_res_df
Xy_regression = read_Xy_4_RRR(verbose=True)
data_df = load_df_from_Xy_regression_setup(['mfr_task'], Xy_regression)
RRR_res_df = RRR_res_df.merge(data_df, left_on=['eid', 'ni'], right_on=['eid', 'ni'])

# compute the R2 of a baseline model that does not incorporate variables as inputs
# that is, it predicts with the mean activity of the neuron averaged across the training set
RRR_res_df = trial_avg_r2(Xy_regression, RRR_res_df)

### save the trained model in the dataFrame format
RRR_res_df[['eid', 'ni', 'uuids', 
            'acronym', 'mfr_task', 
            'RRR_beta', 'RRR_U', 'RRR_V', 'RRR_b', 
            'RRR_r2', 'null_r2',]].to_json(res_fname, index=False)

