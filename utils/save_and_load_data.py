
import pickle, os
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy import signal

from utils.utils import log_kv, make_folder

def get_source_data_folder():
    return './data/downloaded'

def get_processed_data_folder():
    return make_folder('./data/processed')

"""
save and load data
"""
def read_Xy_4_RRR(verbose=False):
    def _load_one_eid(eid):
        data_fname = os.path.join(data_source_folder, f"data_{eid}.npz")
        neural_fname = beh_fname = data_fname
        fname2 = os.path.join(data_2_folder, f"Xy_regression_{eid}.pkl")
        if os.path.isfile(fname2): 
            with open(fname2, "rb") as f:
                Xy_regression_eid = pickle.load(f)
        else:    
            Xy_regression_eid = _read_Xy(neural_fname, beh_fname, eid, 
                                                verbose=verbose)
            with open(fname2, "wb") as f:
                pickle.dump(Xy_regression_eid, f)
        return Xy_regression_eid
    print("start to load data")
    data_2_folder = get_processed_data_folder()
    data_source_folder = get_source_data_folder()

    Xy_regression = {}
    eid_list = list(np.load("./data/eid_list.npy"))
    for eid in tqdm(eid_list):
        eid = str(eid)
        res = _load_one_eid(eid)
        if ("Xall" in res) and ("yall" in res):
            Xy_regression[eid] = res
        else:
            print(f"data for eid {eid} is not valid, skip!")

    print("finish loading data")
    return Xy_regression

    
def _read_Xy(neural_fname, beh_fname, eid, verbose=False):

    def _get_context(task_data, ks_include):
        behavior_all = task_data['behavior'][ks_include]
        context = np.round(np.array([(float(b.split('_')[0]) - 0.5) / 0.3 for b in behavior_all]))  # (-1., 0., 1.)
        return np.repeat(context, T).reshape((K, T, 1))
    def _get_side(task_data, ks_include):
        behavior_all = task_data['behavior'][ks_include]
        sti_lorr = np.round(np.array([float(b.split('_')[1]) for b in behavior_all]))  # (1., -1.)
        return np.repeat(sti_lorr, T).reshape((K, T, 1))
    def _get_contrast(task_data, ks_include):
        behavior_all = task_data['behavior'][ks_include]
        def _b2cont_level(b):
            b = float(b.split('_')[2])
            if b == 0: return 0.
            elif b>=0.25: return 4.
            else: return 1.
        sti_cont_level = np.round(np.array([_b2cont_level(b) for b in behavior_all]))  # (0., 1., 4.)
        return np.repeat(sti_cont_level, T).reshape((K, T, 1))
    def _get_choice(task_data, ks_include):
        behavior_all = task_data['behavior'][ks_include]
        lorr = np.round(np.array([float(b.split('_')[3]) for b in behavior_all]))  # (1., -1.)
        return np.repeat(lorr, T).reshape((K, T, 1))
    def _get_reward(task_data, ks_include):
        behavior_all = task_data['behavior'][ks_include]
        sti_lorr = np.array([float(b.split('_')[1]) for b in behavior_all])  # (1., -1.)
        lorr = np.array([float(b.split('_')[3]) for b in behavior_all])  # (1., -1.)
        reward = np.round((sti_lorr == lorr) * 2 - 1.) # (1., -1.)
        return np.repeat(reward, T).reshape((K, T, 1))
    def _preprocess_movement(_beh_raw):
        _dt = 10
        if len(_beh_raw.shape) == 2:
            _beh_raw = _beh_raw[:,:,np.newaxis]
        # find the best delay
        _delay = []; _success = []; _beh_processed = np.zeros((data.shape[0], data.shape[1], _beh_raw.shape[2]))
        for i in range(_beh_raw.shape[-1]):
            _beh = _beh_raw[:,:,i]
            _bd, _s, _ = find_bestdelay_byCC(_beh, data)
            if _s == False: _bd = _dt
            _delay.append(_bd)
            _success.append(_s)
            _beh_processed[:,:,i] = _beh[:, _bd:_bd + data.shape[1]]  # (K, T)
        _beh_processed = (_beh_processed - np.mean(_beh_processed, axis=(0,1))) / np.std(_beh_processed, axis=(0,1))
        if verbose:
            log_kv(best_delay=_delay, success_best_delay=_success)
        return _beh_processed, _delay, _success
    def _get_wheel(task_data, ks_include):
        print("wheel")
        _beh_processed, _delay, _success = _preprocess_movement(task_data["wheel_vel"][ks_include,:-1])
        best_delay["wheel"] = _delay 
        success_best_delay["wheel"] = _success 
        return _beh_processed
    def _get_lick(task_data, ks_include):
        print("lick")
        _beh_processed, _delay, _success = _preprocess_movement(task_data["licks"][ks_include,:-1])
        best_delay["lick"] = _delay 
        success_best_delay["lick"] = _success 
        return _beh_processed
    def _get_whisker_max(task_data, ks_include):
        _beh_raw = task_data["whisker_motion"][ks_include,:-1]
        _beh_raw = np.max(_beh_raw, -1, keepdims=True)
        _beh_processed, _delay, _success = _preprocess_movement(_beh_raw)
        best_delay["whisker_max"] = _delay 
        success_best_delay["whisker_max"] = _success 
        return _beh_processed
    var2value = {
        "block": _get_context,
        "side": _get_side,
        "contrast_level": _get_contrast, 
        "choice": _get_choice, 
        "outcome": _get_reward,
        "wheel": _get_wheel,
        "lick": _get_lick,
        "whisker_max": _get_whisker_max,}
    
    if verbose:
        print(f"====== {eid} ======")
    result_ret = {}

    task_data = np.load(beh_fname, allow_pickle=True)
    neural_data = np.load(neural_fname, allow_pickle=True)
    ### spike
    spsdt=10e-3
    data_allN = neural_data['spike_count_matrix'][:,10:-11,:]*spsdt  # (K, T, N) # spike count matrix saved firing rates
    data_allN = np.clip(data_allN, 0, None)
    cluster_gs_allN = {}
    for k in neural_data['clusters_g'].item():
        cluster_gs_allN[k] = neural_data['clusters_g'].item()[k]

    ### determine trials to include
    K, T, _ = data_allN.shape
    ks_include = np.ones(data_allN.shape[0], dtype=bool)
    block = np.round(np.array([(float(b.split('_')[0]) - 0.5) / 0.3 for b in task_data['behavior']]))  # (-1., 0., 1.)
    ks_include = ~(block==0.) # remove trials of prob=0.5 block
    data_allN = data_allN[ks_include]

    if K < 100:
        if verbose:
            print(f"remove session due to K{K} < min_trials{100}")
        return result_ret  

    
    ### determine cells to include
    cs = (np.mean(np.all(data_allN == 0., axis=1), axis=0) < 0.5)
    cs &= (np.mean(data_allN, (0, 1))/spsdt > .5)
    good_unit = cluster_gs_allN['label'] >= 0
    cs &= good_unit
    good_area_l = np.unique(cluster_gs_allN['acronym'])
    good_area_l = good_area_l[~np.isin(good_area_l, ['root', 'void', 'y'] )]
    good_area = np.isin(cluster_gs_allN['acronym'], good_area_l)
    cs &= good_area

    data = data_allN[:, :, cs]
    cluster_gs = {}
    for k in cluster_gs_allN:
        cluster_gs[k] = cluster_gs_allN[k][cs]

    K, T, N = data.shape
    if verbose:
        print(f"shape of spike_count_matrix {data.shape}")
    if N < 5:
        if verbose:
            print(f"remove session due to N{N} < min_neurons{5}")
        return result_ret  
    

    ### input variables
    best_delay = {}; success_best_delay = {}; 
    try:
        var_list = ['block', 'side', 'contrast_level', 'choice', "outcome", "wheel", "whisker_max", "lick",]
        F3d = np.concatenate([var2value[var](task_data, ks_include) for var in var_list], -1)  # (K, T, r)
    except:
        if verbose:
            print(f"remove session due to lack of behavioral recordings")
        return result_ret

    ### compute the regression coefficients
    sc_mtx_processed = data.copy()
    # smooth activity
    sc_mtx_processed = gaussian_filter1d(sc_mtx_processed, 2., axis=1)  # (K, T, N)
    # z-score activity for each neuron and each time point
    mean_y = np.mean(sc_mtx_processed, axis=0) # (T, N)
    std_y = np.std(sc_mtx_processed, axis=0) # (T, N)
    std_y = np.clip(std_y, 1e-8, None) # (T, N) 
    sc_mtx_processed = (sc_mtx_processed - mean_y) / std_y

    # z-score inputs for each variable and each time point
    mean_X = np.mean(F3d, axis=0) # (T, v)
    std_X = np.std(F3d, axis=0) # (T, v)
    std_X = np.clip(std_X, 1e-8, None) # (T, N) 
    F3d = (F3d-mean_X) / std_X

    # expand the intercept
    F3d = np.concatenate((F3d, np.ones((K, T, 1))), -1)

    result_ret = {
        "Xall": F3d,
        "yall": sc_mtx_processed,
        'setup': {'best_delay': best_delay, 
                    'success_best_delay': success_best_delay,
                    "mfr_task": np.mean(data, axis=(0,1))/spsdt,
                    "sp_task": np.mean(np.all(data == 0., axis=1), axis=0),
                    "mean_y_TN": mean_y,
                    "std_y_TN": std_y,
                    "mean_X_Tv": mean_X, 
                    "std_X_Tv": std_X,
                    **cluster_gs
                    }
    }
    return result_ret

"""
Function that finds the best delay between behavior and neural activity 
    by maximizing the zero-lag cross-correlation.
"""
def find_bestdelay_byCC(beh, act):
    """
    :beh: (K,T)
    :act: (K,T,N)
    """
    beh = beh - np.mean(beh, axis=1, keepdims=True)
    act = act - np.mean(act, axis=1, keepdims=True)
    lags = signal.correlation_lags(beh.shape[1], act.shape[1], mode="valid")
    CCs = []
    for ni in range(act.shape[-1]):
        CC = np.mean([signal.correlate(beh[k], act[k,:,ni], mode="valid") for k in range(beh.shape[0])], 0)
        CCs.append(CC)
    CCs = np.asarray(CCs)
    CC_agg = np.linalg.norm(CCs, axis=0)
    best_delay = lags[np.argmax(CC_agg)]
    if (best_delay>np.min(lags)) and (best_delay<np.max(lags)):
        success=True
    else:
        best_delay = 0
        success=False
    return best_delay, success, CCs

"""
Function that turns relevant single neuron info 
    stored in the dictionary format (Xy_regression):
        {eid: {"setup": {"uuids": np.array, "acronym": np.array, ...}}, ...}
    into a DataFrame format:
        each row is a neuron, with columns including:
        - eid, ni, uuids: neuron identifier
        - acronym: brain area
        - other requested keys
"""
def load_df_from_Xy_regression_setup(keys, Xy_regression):
    data_dict = dict(eid=[], ni=[], uuids=[], acronym=[])
    for k in keys:
        data_dict[k] = []
    for eid in sorted(list(Xy_regression.keys())):
        N = len(Xy_regression[eid]['setup']["uuids"])
        data_dict['eid'] += [eid] * N
        data_dict['ni'] += list(range(N))
        for k in ['uuids', 'acronym']:
            data_dict[k] += Xy_regression[eid]['setup'][k].tolist()
        for k in keys:
            data_dict[k] += Xy_regression[eid]['setup'][k].tolist()
    data_df = pd.DataFrame.from_dict(data_dict)
    return data_df

"""
Function for unnormalizing the predicted and true firing rates, and return them in the original scale (Hz).
"""
def to_fr(Xy_eids, eid, X, y, y_pred):
    mean_y_TN = Xy_eids[eid]['setup']['mean_y_TN']
    std_y_TN = Xy_eids[eid]['setup']['std_y_TN']
    y_pred = y_pred*std_y_TN + mean_y_TN
    y = y*std_y_TN + mean_y_TN
    X[:,:,:-1] = X[:,:,:-1]*Xy_eids[eid]['setup']['std_X_Tv'] + Xy_eids[eid]['setup']['mean_X_Tv']
    return X, y, y_pred