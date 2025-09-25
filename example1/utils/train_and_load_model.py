
import os
import pandas as pd

from example1.utils.save_and_load_data import get_processed_data_folder
from utils import remove_space, log_kv, make_folder
from RRRGD_main_CV import train_model_hyper_selection
from RRRGD import RRRGD_model


def _get_RRR_fname(data_params, RRRGD_params, spliti=None):
    # save the model in the processed data folder
    # because the resulting model depends on how we process the data
    data_folder = get_processed_data_folder(data_params)
    model_folder = make_folder(os.path.join(data_folder, f"RRRGD_models_{RRRGD_params['id']}"))
    if spliti is None:
        RRRGD_fname = os.path.join(model_folder, remove_space(f"RRRGD_{RRRGD_params}")) 
    else:
        RRRGD_fname = os.path.join(model_folder, remove_space(f"RRRGD_{RRRGD_params}_split{spliti}")) 
    log_kv(RRRGD_fname=RRRGD_fname)
    return RRRGD_fname

def RRRglobal_params(sample=False):
    if sample == False:
        RRRGD_p = dict(n_comp_list=list(range(3,7)), l2_list=[25, 75, 200], 
        # RRRGD_p = dict(n_comp_list=list(range(5,6)), l2_list=[75],  # this is the best
                    stratify_by=[[0,2],[0],None],# first try to stratify by block, contrast_level; if not enough data, stratify by block; if still not enough data, don't stratify
                    nsplit=3, test_size=0.3,
                    history_size=50, max_iter=1000, tolerance_change=1e-4,
                    sample_kwargs=dict(sample=False),)
        RRRGD_p['id'] = "LBFGS"
    else:
        assert False
    return RRRGD_p

"""
train the RRRGD model by considering neurons from *all areas*
- model saved in the data folder: get_data_folder(gp['X_inc'], gp['y_inc'], gp['vl'])
- RRRGD model: 
    - name: RRRGD_{RRRGD_params}.pt
"""
def get_RRRglobal_res(Xy_regression, data_params, RRRGD_params):
    RRRGD_fname = _get_RRR_fname(data_params, RRRGD_params)
    if os.path.isfile(RRRGD_fname):
        pass
    else:
        train_model_hyper_selection(Xy_regression, model_fname=RRRGD_fname,
                                    **RRRGD_params)


def _loadres_RRReids(res_eids, res_dict_temp, mname, spliti=None):
    for eid in sorted(list(res_eids.keys())):
        res = res_eids[eid]
        N = len(res["r2"])
        res_dict_temp['eid'] += [eid] * N
        res_dict_temp['ni'] += list(range(N))
        res_dict_temp[f"{mname}_r2"] += res["r2"].tolist()
        if spliti is None:
            res_dict_temp[f"{mname}_beta"] += [res["model"]["beta"][ni] for ni in range(N)] # (ncoef, T)
        else:
            res_dict_temp[f"{mname}{spliti}_beta"] += [res["model"]["beta"][ni] for ni in range(N)] # (ncoef, T)
        res_dict_temp[f"{mname}_U"] += [res["model"]["U"][ni] for ni in range(N)] # (ncoef, ncomp)
        res_dict_temp[f"{mname}_V"] += [res["model"]["V"]] * N # (ncomp, T)
        res_dict_temp[f"{mname}_b"] += [res["model"]["b"][ni][0] for ni in range(N)] # (T)
    return res_dict_temp



"""
load the RRRGD model of *multiple areas*
"""
def load_RRRglobal_res(data_params, RRRGD_params, spliti=None):
    mname = "RRRglobal"
    if spliti is not None:
        beta = f"{mname}{spliti}_beta"
    else:
        beta = f"{mname}_beta"
        
    columns = ["eid", "ni", beta, f"{mname}_r2"] 
    columns += [f"{mname}_U", f"{mname}_V", f"{mname}_b"]
    res_dict_temp = {c: [] for c in columns}

    RRRGD_fname = _get_RRR_fname(data_params, RRRGD_params, spliti=spliti)
    if os.path.isfile(f"{RRRGD_fname}.pt"):
        res_all = RRRGD_model.params2RRRres(RRRGD_fname) 
    else:
        assert False, f"no file {RRRGD_fname}"

    res_dict_temp = _loadres_RRReids(res_all, res_dict_temp, mname)
    RRR_res_df = pd.DataFrame.from_dict(res_dict_temp)

    return RRR_res_df
