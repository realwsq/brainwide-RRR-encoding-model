from example1.utils.utils import import_areagroup, p_to_text
from example1.utils.save_and_load_data import read_Xy_encoding2, get_data_folder
from example1.utils.analyze_coef_utils import load_timescale_neuron
from example1.utils.import_head import global_params
from utils import make_folder

import os, pdb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


gp_setup = dict(wa = 'cortexbwm', vt='clean', it=f"original")
gp = global_params(which_areas=gp_setup['wa'], var_types=gp_setup['vt'], inc_type=gp_setup['it'])
local_folder_lf = make_folder(get_data_folder("RRR_local_folder_original"))
resgood_folder = make_folder("./example1/res_good")
RRR_res_df = pd.read_json(os.path.join(local_folder_lf, "RRRglobal_full.json"))



res = import_areagroup()
conn_area_list_byH = res['cortical_area_list']
moduleH2li = {"visual": 0, "somatomotor": 1, "auditory":2, "lateral": 3, "medial": 4, "prefrontal": 5}
H1 = res['hierarchy'][0]
area2H = {H1[_H][0]: _H for _H in H1}
area2ci_byHarris = res['hierarchy_byHarris'][1]


inc_param=dict(min_N=50, min_r2=0.015) 
RRR_res_df['RRRglobal_r2inc'] = RRR_res_df["RRRglobal_r2"] - RRR_res_df['meanact_r2']
nis_incmask = RRR_res_df['RRRglobal_r2inc'] > inc_param['min_r2']
nis_incmask_ctx = nis_incmask & RRR_res_df.acronym.isin(conn_area_list_byH)
# sort areas
area_order_H = np.array([a for a in conn_area_list_byH if np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=inc_param['min_N']])


### signal tau increases with hierarchy
Xy_regression = read_Xy_encoding2(gp, verbose=True)
RRR_res_df = load_timescale_neuron(Xy_regression, RRR_res_df, mname='RRRglobal', local_folder=local_folder_lf, delay_max=None)
perc_mean = lambda a, yv: np.mean(RRR_res_df.loc[(nis_incmask_ctx)&(RRR_res_df.acronym==a), yv])
_toplot = [f'signal_tau_neuron RRRglobal_None', 'mfr_task']
fig, axes = plt.subplots(len(_toplot), 1, figsize=(9., 3.5*len(_toplot)))
for ri, k in enumerate(_toplot):
    if k == "RRRglobal_r2inc": k_display = r"$\Delta R^2$"
    elif k == "signal_tau_neuron RRRglobal_None": k_display = "signal timescale (s)"
    elif k == "all_tau_neuron RRRglobal_None": k_display = "all timescale (s)"
    elif k == "mfr_task": k_display = "mean firing rate (Hz)"
    
    results = {'region': np.array([a for a in area_order_H]), 
            k_display: np.array([perc_mean(a, k) for a in area_order_H]),
            "hierarchy": np.array([area2H[a] for a in area_order_H]), 
            "li": np.array([moduleH2li[area2ci_byHarris[a]] for a in area_order_H])}
    pd.DataFrame().from_dict(results).to_csv(os.path.join(local_folder_lf, f"areabyarea_hierarchy_major_wCI_{inc_param.values()}.csv"))
    ax = axes[ri]
    temp = RRR_res_df[(nis_incmask_ctx)&(RRR_res_df.acronym.isin(area_order_H))][['acronym', k]]
    temp['hierarchy'] = [area2H[a] for a in temp.acronym]
    sns.regplot(data=temp, x='hierarchy', y=k, x_estimator=np.mean, n_boot=5000, ax=ax)
    for i in range(len(results['region'])):
        ax.text(results['hierarchy'][i], results[k_display][i], results['region'][i],
                size=8, zorder=2, color='w', fontweight='bold')
        c = plt.get_cmap('tab10')(results["li"][i])
        ax.text(results['hierarchy'][i], results[k_display][i], results['region'][i],
                    size=9, zorder=3, color=c)
    r, p = pearsonr(results['hierarchy'], results[k_display])
    rs, ps = spearmanr(results['hierarchy'], results[k_display])
    print(k_display)
    print("r: ", r, "p: ", p)
    print("rs: ", rs, "ps: ", ps)
    ax.set_title(f'{p_to_text(ps)}  '+r"$\rho$="+f"{rs:.2f}")
    ax.set_xlabel('Position in the hierarchy')
    ax.set_ylabel(k_display)

plt.tight_layout(); sns.despine(); 
plt.savefig(os.path.join(resgood_folder, f"areabyarea_hierarchy_major_wCI_{inc_param.values()}.pdf")); plt.close('all')
pdb.set_trace()
