
import pickle, os
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from utils import log_kv, remove_space, make_folder
from example1.utils.utils import find_bestdelay_byCC


def get_data_folder(*args):
    subfolder = ""
    for arg in args:
        subfolder += f"{arg}"
    data_folder = make_folder(os.path.join(f"/burg/stats/users/sw3894/bwm_original/cortexbwm/", remove_space(subfolder)))
    return data_folder


"""
save and load data
"""
def read_Xy_encoding2(gp, 
                      verbose=False):
    def _load_one_eid(eid):
        fname2 = os.path.join(data_2_folder, remove_space(f"Xy_regression_wtonguepaw_{eid}_all.pkl"))
        if os.path.isfile(fname2): 
            with open(fname2, "rb") as f:
                Xy_regression_eid = pickle.load(f)
        else:    
            Xy_regression_eid = read_Xy_encoding(gp['vl'], neural_fname, beh_fname, eid, 
                                                exclude_areas=gp['al_exclude'] if 'al_exclude' in gp else None,
                                                include_areas=gp['al_include'] if 'al_include' in gp else None,
                                                **gp['X_inc'], **gp['y_inc'],
                                                verbose=verbose)
            with open(fname2, "wb") as f:
                pickle.dump(Xy_regression_eid, f)
        return Xy_regression_eid
    print("start to load data")
    data_2_folder = get_data_folder(gp['X_inc'], gp['y_inc'], gp['vl'])
    data_source_folder = get_data_folder("downloaded")

    Xy_regression = {}
    eid_list = np.load("example1/utils/eid_list.npy")
    for eid in tqdm(eid_list):
        data_fname = os.path.join(data_source_folder, f"data_wtonguepaw_{eid}_all_spsT10_False.npz")
        neural_fname = beh_fname = data_fname
        Xy_regression_eid = _load_one_eid(eid)
        if "all" in Xy_regression_eid:
            Xy_regression[eid] = Xy_regression_eid["all"]
        else:
            print(f"no data in the eid {eid}")
            pass
    print("finish loading data")
    return Xy_regression

    
def read_Xy_encoding(var_list, neural_fname, beh_fname, eid, 
                     smooth_w=1e-3, min_mfr=1, max_sp=1., transform_mfr=None, standardize_y=True, min_trials=1, min_neurons=1,unit_label_min=0.,
                     shift_beh=True, remove_block5=False, standardize_X=False,
                     exclude_areas=None, include_areas=None, 
                     spsdt=10e-3, verbose=False):
    if verbose:
        print(f"====== {eid} ======")
    result_ret = {}


    def _get_context(_temp, ks_include):
        behavior_all = _temp['behavior'][ks_include]
        context = np.round(np.array([(float(b.split('_')[0]) - 0.5) / 0.3 for b in behavior_all]))  # (-1., 0., 1.)
        return np.repeat(context, T).reshape((K, T, 1))
    def _get_side(_temp, ks_include):
        behavior_all = _temp['behavior'][ks_include]
        sti_lorr = np.round(np.array([float(b.split('_')[1]) for b in behavior_all]))  # (1., -1.)
        return np.repeat(sti_lorr, T).reshape((K, T, 1))
    def _get_contrast(_temp, ks_include):
        behavior_all = _temp['behavior'][ks_include]
        def _b2cont_level(b):
            b = float(b.split('_')[2])
            if b == 0: return 0.
            elif b>=0.25: return 4.
            else: return 1.
        sti_cont_level = np.round(np.array([_b2cont_level(b) for b in behavior_all]))  # (0., 1., 4.)
        return np.repeat(sti_cont_level, T).reshape((K, T, 1))
    def _get_choice(_temp, ks_include):
        behavior_all = _temp['behavior'][ks_include]
        lorr = np.round(np.array([float(b.split('_')[3]) for b in behavior_all]))  # (1., -1.)
        return np.repeat(lorr, T).reshape((K, T, 1))
    def _get_reward(_temp, ks_include):
        behavior_all = _temp['behavior'][ks_include]
        sti_lorr = np.array([float(b.split('_')[1]) for b in behavior_all])  # (1., -1.)
        lorr = np.array([float(b.split('_')[3]) for b in behavior_all])  # (1., -1.)
        reward = np.round((sti_lorr == lorr) * 2 - 1.) # (1., -1.)
        return np.repeat(reward, T).reshape((K, T, 1))
    def _preprocess_movement(_beh_raw):
        _dt = 10
        if len(_beh_raw.shape) == 2:
            _beh_raw = _beh_raw[:,:,np.newaxis]
        if not shift_beh:
            _beh_processed = _beh_raw[:,_dt:_dt+data.shape[1],:]  # _dt time bins delay besigned when downloading data
            _delay = None; _success = None # placeholder
        else:
            # find the best delay
            _delay = []; _success = []; _beh_processed = np.zeros((data.shape[0], data.shape[1], _beh_raw.shape[2]))
            for i in range(_beh_raw.shape[-1]):
                _beh = _beh_raw[:,:,i]
                _bd, _s, _ = find_bestdelay_byCC(_beh, data, plot=False)
                if _s == False: _bd = _dt
                _delay.append(_bd)
                _success.append(_s)
                _beh_processed[:,:,i] = _beh[:, _bd:_bd + data.shape[1]]  # (K, T)
        _beh_processed = (_beh_processed - np.mean(_beh_processed, axis=(0,1))) / np.std(_beh_processed, axis=(0,1))
        if verbose:
            log_kv(best_delay=_delay, success_best_delay=_success)
        return _beh_processed, _delay, _success
    def _get_wheel(_temp, ks_include):
        print("wheel")
        _beh_processed, _delay, _success = _preprocess_movement(_temp["wheel_vel"][ks_include,:-1])
        best_delay["wheel"] = _delay 
        success_best_delay["wheel"] = _success 
        return _beh_processed
    def _get_lick(_temp, ks_include):
        print("lick")
        _beh_processed, _delay, _success = _preprocess_movement(_temp["licks"][ks_include,:-1])
        best_delay["lick"] = _delay 
        success_best_delay["lick"] = _success 
        return _beh_processed
    def _get_whisker_max(_temp, ks_include):
        _beh_raw = _temp["whisker_motion"][ks_include,:-1]
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
        
    _temp = np.load(beh_fname, allow_pickle=True)
    _temp_neural = np.load(neural_fname, allow_pickle=True)
    ### spike
    data_allN = _temp_neural['spike_count_matrix'][:,10:-11,:]*spsdt  # (K, T, N) # spike count matrix saved firing rates
    data_allN = np.clip(data_allN, 0, None)
    cluster_gs_allN = {}
    for k in _temp_neural['clusters_g'].item():
        cluster_gs_allN[k] = _temp_neural['clusters_g'].item()[k]

    ### determine trials to include
    K, T, _ = data_allN.shape
    ks_include = np.ones(data_allN.shape[0], dtype=bool)
    if remove_block5:
        block = np.round(np.array([(float(b.split('_')[0]) - 0.5) / 0.3 for b in _temp['behavior']]))  # (-1., 0., 1.)
        ks_include = ~(block==0.)
    data_allN = data_allN[ks_include]

    if K < min_trials:
        if verbose:
            print(f"remove session due to K{K} < min_trials{min_trials}")
        return result_ret  

    
    ### determine cells to include
    cs = (np.mean(np.all(data_allN == 0., axis=1), axis=0) < max_sp)
    cs &= (np.mean(data_allN, (0, 1))/spsdt > min_mfr)
    if verbose:
        print(f"{np.mean(cs)} neuron with silent prob < {max_sp}")
    good_unit = cluster_gs_allN['label'] >= unit_label_min
    if verbose:
        print(f"{np.mean(good_unit)} neuron with label >= {unit_label_min}")
    cs &= good_unit
    if include_areas is None:
        good_area_l = np.unique(cluster_gs_allN['acronym'])
    else:
        good_area_l = np.asarray(include_areas)
    if (exclude_areas is not None) and (type(exclude_areas) is list) and (len(exclude_areas) > 0):
        good_area_l = good_area_l[~np.isin(good_area_l, exclude_areas)]
    good_area = np.isin(cluster_gs_allN['acronym'], good_area_l)
    cs &= good_area

    data = data_allN[:, :, cs]
    cluster_gs = {}
    for k in cluster_gs_allN:
        cluster_gs[k] = cluster_gs_allN[k][cs]

    K, T, N = data.shape
    if verbose:
        print(f"shape of spike_count_matrix {data.shape}")
    if N < min_neurons:
        if verbose:
            print(f"remove session due to N{N} < min_neurons{min_neurons}")
        return result_ret  
    

    ### input variables
    best_delay = {}; success_best_delay = {}; 
    try:
        F3d = np.concatenate([var2value[var](_temp, ks_include) for var in var_list], -1)  # (K, T, r)
    except:
        return result_ret

    ### compute the regression coefficients
    sc_mtx_processed = data.copy()
    # transform activity for each neuron
    if transform_mfr is None:
        pass
    elif transform_mfr == "sqrt":
        sc_mtx_processed = np.sqrt(sc_mtx_processed)
    elif transform_mfr == "log":
        sc_mtx_processed = np.log(sc_mtx_processed + 1e-3)
    # smooth activity
    if smooth_w > 0:
        sc_mtx_processed = gaussian_filter1d(sc_mtx_processed, smooth_w, axis=1)  # (K, T, N)
    else:
        sc_mtx_processed = data.copy()

    if standardize_y:
        # z-score activity for each neuron and each time point
        mean_y = np.mean(sc_mtx_processed, axis=0) # (T, N)
        std_y = np.std(sc_mtx_processed, axis=0) # (T, N)
        std_y = np.clip(std_y, 1e-8, None) # (T, N) 
    else:
        mean_y = np.zeros(sc_mtx_processed.shape[1:])
        std_y = np.ones(sc_mtx_processed.shape[1:])
    sc_mtx_processed = (sc_mtx_processed - mean_y) / std_y

    if standardize_X:
        # z-score inputs for each variable and each time point
        mean_X = np.mean(F3d, axis=0) # (T, v)
        std_X = np.std(F3d, axis=0) # (T, v)
        std_X = np.clip(std_X, 1e-8, None) # (T, N) 
    else:
        mean_y = np.zeros(F3d.shape[1:])
        std_y = np.ones(F3d.shape[1:])
    F3d = (F3d-mean_X) / std_X

    # expand the intercept
    F3d = np.concatenate((F3d, np.ones((K, T, 1))), -1)

    result_ret["all"] = {
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



def iterate_over_eids(Xy_regression, _main_f, **kwargs):
    for eid in tqdm(Xy_regression, desc='iterating over Xy_regression eids'):
        _main_f(Xy_regression, eid, **kwargs)


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
