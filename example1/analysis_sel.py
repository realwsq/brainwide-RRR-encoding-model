from example1.utils.utils import import_areagroup, p_to_text
from example1.utils.save_and_load_data import get_data_folder
from example1.utils.import_head import global_params
from utils import make_folder

import os, pdb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import mahalanobis
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
[conn_mat, conn_i2a, conn_a2i] = res['conn_mat']
dis_mat = 1/np.log(conn_mat)


inc_param=dict(min_N=20, min_r2=0.015) 
RRR_res_df['RRRglobal_r2inc'] = RRR_res_df["RRRglobal_r2"] - RRR_res_df['meanact_r2']
nis_incmask = RRR_res_df['RRRglobal_r2inc'] > inc_param['min_r2']
nis_incmask_ctx = nis_incmask & RRR_res_df.acronym.isin(conn_area_list_byH)
# sort areas
area_order_H = np.array([a for a in conn_area_list_byH if np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=inc_param['min_N']])
area_order_moduleH = area_order_H[np.argsort(np.array([(moduleH2li[area2ci_byHarris[a]], area2H[a]) for a in area_order_H], dtype=[('x', int),('y', int)]), order=('x', 'y'))]

### selectivity similarity decreases with connectivity
coef_vs_all = np.array([_[:-1] for _ in RRR_res_df.loc[nis_incmask_ctx,"RRRglobal_beta"]])
sel_total = np.abs(coef_vs_all).sum(2)
sel_areas = np.asarray([sel_total[RRR_res_df.loc[nis_incmask_ctx,"acronym"]==a].mean(0) for a in area_order_moduleH])
sel_areas = (sel_areas - np.mean(sel_areas,0))/np.std(sel_areas, 0) # z-score per input variable
np.savez(os.path.join(local_folder_lf, f"selnormed_areas_{inc_param.values()}.npz"), sel_areas=sel_areas, area_order_moduleH=area_order_moduleH, vl=gp['vl'])


### plot the 8-D selectivity profile
fig, ax = plt.subplots(1,1,figsize=(6.,3.5))
improp = dict(aspect='auto', cmap='Greys', interpolation='nearest', 
            vmax=np.max(sel_areas), vmin=np.min(sel_areas))
im = ax.imshow(sel_areas.T, **improp)
for vi in range(len(gp['vl'])):
    _sorted = np.argsort(sel_areas[:,vi])[::-1]
    _n = 3
    ais = _sorted[:_n]
    for ai in ais:
        x,y = ai, vi+0.5
        _ = ax.annotate("*", xy=(x, y), 
                        xytext=(x, y),  
                        ha='center', 
                        va='bottom',
                        c='r')
# setup of labels
_=ax.set_yticks(np.arange(sel_areas.shape[1]))
_=ax.set_yticklabels(gp['vl'])
# _=ax.set_yticklabels(gp['vl'])
_=ax.set_xticks(np.arange(len(area_order_moduleH)))
_=ax.set_xticklabels(area_order_moduleH, rotation = 90, ha="right")
[t.set_color(i) for (i,t) in zip([plt.get_cmap('tab10')(moduleH2li[area2ci_byHarris[a]]) for a in area_order_moduleH], ax.xaxis.get_ticklabels())]
# setup of colorbar
cbar = plt.colorbar(im, orientation="horizontal", pad=.25, shrink=.4); 
_=cbar.ax.set_xlabel(r'selectivity $\alpha_a^v$')
sns.despine(); plt.tight_layout(); plt.savefig(os.path.join(resgood_folder, f"selnormed_{inc_param.values()}_vHarris.pdf")); plt.close('all')


# similarity of sel vs anatomical distance
inc_param=dict(min_N=50, min_r2=0.015) 
nis_incmask = RRR_res_df['RRRglobal_r2inc'] > inc_param['min_r2']
nis_incmask_ctx = nis_incmask & RRR_res_df.acronym.isin(conn_area_list_byH)
# sort areas
area_order_H = np.array([a for a in conn_area_list_byH if (np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=inc_param['min_N'])])

### selectivity similarity decreases with connectivity
coef_vs_all = np.array([_[:-1] for _ in RRR_res_df.loc[nis_incmask_ctx,"RRRglobal_beta"]])
sel_total = np.abs(coef_vs_all).sum(2)
sel_areas = np.asarray([sel_total[RRR_res_df.loc[nis_incmask_ctx,"acronym"]==a].mean(0) for a in area_order_H])
sel_areas = (sel_areas - np.mean(sel_areas,0))/np.std(sel_areas, 0) # z-score per input variable
np.savez(os.path.join(local_folder_lf, f"selnormed_areas_{inc_param.values()}.npz"), sel_areas=sel_areas, area_order_H=area_order_H, vl=gp['vl'])

area_pair_res = dict(dis=[], conn=[], pairs=[], sim=[])
for i in range(len(area_order_H)):
    for j in range(i+1, len(area_order_H)):
        area_pair_res['dis'].append(dis_mat[conn_a2i[area_order_H[i]], conn_a2i[area_order_H[j]]])
        area_pair_res['conn'].append(1/dis_mat[conn_a2i[area_order_H[i]], conn_a2i[area_order_H[j]]])
        a = sel_areas[i]; b = sel_areas[j]
        area_pair_res['sim'].append(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
        area_pair_res['pairs'].append([area_order_H[i], area_order_H[j]])
pd.DataFrame().from_dict(area_pair_res).to_csv(os.path.join(local_folder_lf, f"selnormed_areapairs_hierarchy_major_wCI_{inc_param.values()}.csv"))

fig, ax = plt.subplots(1,1,figsize=(5., 3.5))
rs, ps = spearmanr(area_pair_res['conn'], area_pair_res['sim'])
sns.regplot(x=area_pair_res['conn'], y=area_pair_res['sim'], ax=ax)
ax.set_ylabel("cosine similarty")
ax.set_xlabel(r"log($C_{ij}$)")
ax.set_title(f'{p_to_text(ps)}  '+r"$\rho$="+f"{rs:.2f}")
# add some notations for the pairs
pair_is = np.where((np.array(area_pair_res['sim'])>0.9))[0]
for ri in pair_is:
    x = area_pair_res['conn'][ri]
    y = area_pair_res['sim'][ri]
    ax.text(x=x, y=y, s=area_pair_res['pairs'][ri],
            fontsize=8, color='black', ha='center', va='center')

sns.despine(); plt.tight_layout()
plt.savefig(os.path.join(resgood_folder, f"selnormed_dis_{inc_param.values()}.pdf")); plt.close('all')



pdb.set_trace()


