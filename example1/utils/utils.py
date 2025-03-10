import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def load_neuron_nparray(df, column, rows_mask=None):
    if rows_mask is None:
        vs = df[column]
    else:
        vs = df.loc[rows_mask, column]
    vs = np.array([v for v in vs])
    return vs


def find_bestdelay_byCC(beh, act, plot=False):
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
    if plot:
        plt.figure(figsize=(3,3))
        plt.plot(lags, CC_agg)
        plt.axvline(x=best_delay)
        plt.show()
    return best_delay, success, CCs


# ### compare the single-trial performance
def plot_r2_comp(RRR_res_df, nis_incmask, mname1, mname1_disp,
                 resgood_folder):
    mname2 = "RRRglobal"
    fig, axes = plt.subplots(1,2,figsize=(3.*2, 3.))
    ax = axes[0]
    ax.scatter(RRR_res_df.loc[nis_incmask, f"{mname1}_r2"], RRR_res_df.loc[nis_incmask, f"{mname2}_r2"], s=3, alpha=0.5)
    ax.plot([0,0.7], [0,0.7], c='k', linestyle='--')
    ax.set_xlabel(r"$R^2$ of "+mname1_disp)
    ax.set_ylabel(r"$R^2$ of reduced-rank regression")
    ax = axes[1]
    _deltas = RRR_res_df.loc[nis_incmask, f"{mname2}_r2"]-RRR_res_df.loc[nis_incmask, f"{mname1}_r2"]
    ax.hist(_deltas, bins=50, log=True)
    ax.axvline(x=0., c='k', linestyle='-')
    _mean = np.mean(_deltas)
    # ax.axvline(x=_mean, c='r', linestyle='-')
    ax.set_xlabel(r"$\Delta R^2$")
    ax.set_ylabel("number of neurons")
    plt.tight_layout(); sns.despine()
    plt.savefig(os.path.join(resgood_folder, f"R2_{mname2}_{mname1}.pdf")); plt.close('all')


def p_to_text(p):
    if p < 0.0001:
        return '*** P=%.1e' % p
    if p < 0.001:
        return '*** P=%.4f' % p
    if p < 0.01:
        return '** P=%.3f' % p
    if p < 0.05:
        return '* P=%.3f' % p
    if p >= 0.05:
        return 'ns P=%.2f' % p


def get_area_anatomical_info_Allen():

    ## Hierarchy 
    conn_area_list = pd.read_csv('./example1/utils/area_list.csv', header=None).values[:,0]
    conn_i2a = {i: a for i, a in enumerate(conn_area_list)}
    conn_a2i = {a: i for i, a in enumerate(conn_area_list)}


    ## area2area connectivity
    conn_mat = pd.read_csv('./example1/utils/conn_cxcx.csv', header=None)
    conn_mat = conn_mat.values


    ## Module 
    mod2area = {
        "prefrontal": ["FRP", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl"], 
        "lateral": ["AId", "AIv", "AIp", "GU", "VISC", "TEa", "PERI", "ECT"], 
        "somatomotor": ["SSs", "SSp-bfd", "SSp-tr", "SSp-ll", "SSp-ul", "SSp-un", "SSp-n", "SSp-m", "MOp", "MOs", ], 
        "visual": ["VISal", "VISl", "VISp", "VISpl", "VISli", "VISpor", "VISrl"], 
        "medial": ["VISa", "VISam", "VISpm", "RSPagl", "RSPd", "RSPv"], 
        "auditory": ["AUDd", "AUDp", "AUDpo", "AUDv"], 
    }
    area2mod = {}
    for mod in mod2area:
        for a in mod2area[mod]:
            area2mod[a] = mod

    return {"cortical_area_list": conn_area_list, 
            "conn_mat": conn_mat, 
            "area2H": conn_a2i,
            "H2area": conn_i2a,
            "mod2area": mod2area,
            "area2mod": area2mod}