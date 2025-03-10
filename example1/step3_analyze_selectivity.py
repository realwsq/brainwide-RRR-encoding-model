from example1.utils.utils import get_area_anatomical_info_Allen, p_to_text

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns



# folder where the trained model is saved
local_folder_lf = "./example1/trained_model"
# load the trained model
RRR_res_df = pd.read_json(os.path.join(local_folder_lf, "RRRglobal_full.json"))
# input variables used in the model
vars_list = ['block', 'side', 'contrast_level', 'choice', "outcome", "wheel", "whisker_max", "lick"]
# folder where the results will be saved
resgood_folder = "./example1/results"

# load the anatomical information that will be used later
area_atm_info = get_area_anatomical_info_Allen()
conn_area_list_byH = area_atm_info['cortical_area_list'] # list of cortical areas, ordered by hierarchy
conn_mat = np.log(area_atm_info['conn_mat']) # pairwise distance matrix
area2mod = area_atm_info['area2mod'] # area to module mapping
area2H = area_atm_info['area2H'] # area to hierarchy mapping



# only include neurons whose R2 pass the minimmum \Delta R2 threshold
#                       and of the cortical areas
inc_param=dict(min_N=20, min_r2=0.015)  # here we lower min_N from 50 to 20 to show selectivity for more areas
RRR_res_df['RRRglobal_r2inc'] = RRR_res_df["RRRglobal_r2"] - RRR_res_df['meanact_r2']
nis_incmask = RRR_res_df['RRRglobal_r2inc'] > inc_param['min_r2']
nis_incmask_ctx = nis_incmask & RRR_res_df.acronym.isin(conn_area_list_byH)
# sort areas by hierarchy
area_order_H = np.array([a for a in conn_area_list_byH if np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=inc_param['min_N']])
# sort areas first by module and then by hierarchy
mod2mi = {"visual": 0, "somatomotor": 1, "auditory":2, "lateral": 3, "medial": 4, "prefrontal": 5}
area_order_moduleH = area_order_H[np.argsort(np.array([(mod2mi[area2mod[a]], area2H[a]) for a in area_order_H], dtype=[('x', int),('y', int)]), order=('x', 'y'))][::-1]


# get the mean abs. selectivity per brain region
coef_vs_all = np.array([_[:-1] for _ in RRR_res_df.loc[nis_incmask_ctx,"RRRglobal_beta"]]) # (n_neurons, n_vars, n_timesteps)
sel_total = np.abs(coef_vs_all).sum(2) 
sel_areas = np.asarray([sel_total[RRR_res_df.loc[nis_incmask_ctx,"acronym"]==a].mean(0) for a in area_order_moduleH])
sel_areas = (sel_areas - np.mean(sel_areas,0))/np.std(sel_areas, 0) # z-score per input variable


### plot the 8-D selectivity profile
fig, ax = plt.subplots(1,1,figsize=(3.5, 7))
improp = dict(aspect='auto', cmap='Greys', interpolation='nearest', 
            vmax=np.max(sel_areas), vmin=np.min(sel_areas))
im = ax.imshow(sel_areas, **improp)
for vi in range(len(vars_list)):
    _sorted = np.argsort(sel_areas[:,vi])[::-1]
    _n = 3
    ais = _sorted[:_n]
    for ai in ais:
        y,x = ai+0.5, vi
        _ = ax.annotate("*", xy=(x, y), 
                        xytext=(x, y),  
                        ha='center', 
                        va='bottom',
                        c='r')
# setup of labels
_=ax.set_xticks(np.arange(sel_areas.shape[1]))
_=ax.set_xticklabels(vars_list, rotation = 90) 
_=ax.set_yticks(np.arange(len(area_order_moduleH)))
_=ax.set_yticklabels(area_order_moduleH) 
[t.set_color(i) for (i,t) in zip([plt.get_cmap('tab10')(mod2mi[area2mod[a]]) for a in area_order_moduleH], ax.yaxis.get_ticklabels())]
# setup of colorbar
cbar = plt.colorbar(im, shrink=.4); 
_=cbar.ax.set_ylabel(r'mean $\alpha_a$ (z-scored)')
sns.despine(); plt.tight_layout(); plt.savefig(os.path.join(resgood_folder, f"mean_abs_sel_per_area.pdf")); plt.close('all')



### correlate the similarity of selectivity with the anatomical distance
# only include areas with more than 50 neurons
inc_param['min_N']=50
area_order_H = np.array([a for a in conn_area_list_byH if (np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=inc_param['min_N'])])
coef_vs_all = np.array([_[:-1] for _ in RRR_res_df.loc[nis_incmask_ctx,"RRRglobal_beta"]])
# get the mean abs. selectivity per brain region
sel_total = np.abs(coef_vs_all).sum(2)
sel_areas = np.asarray([sel_total[RRR_res_df.loc[nis_incmask_ctx,"acronym"]==a].mean(0) for a in area_order_H])
sel_areas = (sel_areas - np.mean(sel_areas,0))/np.std(sel_areas, 0) 



area_pair_res = dict(conn=[], pairs=[], sim=[])
for i in range(len(area_order_H)):
    for j in range(i+1, len(area_order_H)):
        area_pair_res['conn'].append(conn_mat[area2H[area_order_H[i]], area2H[area_order_H[j]]])
        a = sel_areas[i]; b = sel_areas[j]
        area_pair_res['sim'].append(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
        area_pair_res['pairs'].append([area_order_H[i], area_order_H[j]])

fig, ax = plt.subplots(1,1,figsize=(3.5, 3.5))
rs, ps = spearmanr(area_pair_res['conn'], area_pair_res['sim'])
sns.regplot(x=area_pair_res['conn'], y=area_pair_res['sim'], ax=ax)
ax.set_ylabel("Selectivity similarty")
ax.set_xlabel("Anatomical connectivity")
ax.set_title(f'{p_to_text(ps)}  '+r"$\rho$="+f"{rs:.2f}")
# add some notations for the pairs
pair_is = np.where((np.array(area_pair_res['sim'])>0.9))[0]
for ri in pair_is:
    x = area_pair_res['conn'][ri]
    y = area_pair_res['sim'][ri]
    ax.text(x=x, y=y, s=area_pair_res['pairs'][ri],
            fontsize=8, color='black', ha='center', va='center')

sns.despine(); plt.tight_layout()
plt.savefig(os.path.join(resgood_folder, f"sel_sim_vs_atm_conn.pdf")); plt.close('all')




