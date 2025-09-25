from example1.utils.utils import get_area_anatomical_info_Allen, p_to_text
from example1.utils.save_and_load_data import read_Xy_encoding2, data_params, get_data_folder
from example1.utils.analyze_coef_utils_backup import load_timescale_neuron
from utils import make_folder

import os, pdb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


large_lf_folder = make_folder(os.path.join(get_data_folder('all_but_invalid'), "large_lf"))
# load the trained model
RRR_res_df = pd.read_json("./example1/trained_model/RRRglobal_full.json")
# input variables used in the model
vars_list = ['block', 'side', 'contrast', 'choice', "outcome", "wheel", "whisker", "lick"]
# folder where the results will be saved
resgood_folder = make_folder("./example1/results")

# load the anatomical information that will be used later
area_atm_info = get_area_anatomical_info_Allen()
conn_area_list_byH = area_atm_info['cortical_area_list'] # list of cortical areas, ordered by hierarchy
moduleH2li = {"visual": 0, "somatomotor": 1, "auditory":2, "lateral": 3, "medial": 4, "prefrontal": 5}
area2H = area_atm_info['area2H']
area2ci_byHarris = area_atm_info['area2mod']


# only include neurons whose R2 pass the minimmum \Delta R2 threshold
#                       and of the cortical areas
inc_param=dict(min_N=50, min_deltaR2=0.015)  
RRR_res_df['RRRglobal_deltaR2'] = RRR_res_df["RRRglobal_r2"] - RRR_res_df['meanact_r2']
nis_incmask = RRR_res_df['RRRglobal_deltaR2'] > inc_param['min_deltaR2']
nis_incmask_ctx = nis_incmask & RRR_res_df.acronym.isin(conn_area_list_byH)
# sort areas by hierarchy
area_order_H = np.array([a for a in conn_area_list_byH if np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=inc_param['min_N']])

### signal tau increases with hierarchy
data_p = data_params(which_areas='all_but_invalid')
Xy_regression = read_Xy_encoding2(data_p, verbose=True)
RRR_res_df = load_timescale_neuron(Xy_regression, RRR_res_df, mname='RRRglobal', local_folder=large_lf_folder, delay_max=None)
perc_mean = lambda a, yv: np.mean(RRR_res_df.loc[(nis_incmask_ctx)&(RRR_res_df.acronym==a), yv])

fig, ax = plt.subplots(1, 1, figsize=(9., 3.5*1))
k = "signal_tau_neuron RRRglobal_None"; k_display = "signal timescale (s)"
results = {'region': np.array([a for a in area_order_H]), 
        k_display: np.array([perc_mean(a, k) for a in area_order_H]),
        "hierarchy": np.array([area2H[a] for a in area_order_H]), 
        "li": np.array([moduleH2li[area2ci_byHarris[a]] for a in area_order_H])}
pd.DataFrame().from_dict(results).to_csv(os.path.join(resgood_folder, f"areabyarea_hierarchy_major_wCI_{inc_param.values()}.csv"))
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