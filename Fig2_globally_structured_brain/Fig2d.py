###########################################################################
### Supplementary code for the paper:
### "Rarely categorical, highly separable representations along the cortical hierarchy"
### L. Posani*, S. Wang*, S. Muscinalli, L. Paninski$, and S. Fusi$ (2026).
### Relevant: Fig.2d (Correlation between region-to-region functional similarity and their anatomical connectivity)
###########################################################################

### Import packages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

### Prepare data for Fig.2d
###     Data include the selectivity similarity and anatomical connectivity for all pairs of cortical areas (area_pair_res)
# load the trained model
RRR_res_df = pd.read_json("../trained_RRR_model/RRR_selectivity.json")

# only include neurons 
#   1. whose R2 pass the minimmum \Delta R2 threshold (0.015)
#   2. that are located in cortical areas
ctx_area_list = pd.read_csv('../data/area_list.csv', header=None).values[:,0]  # list of cortical areas, ordered by hierarchy
RRR_res_df['RRR_deltaR2'] = RRR_res_df["RRR_r2"] - RRR_res_df['null_r2']
nis_incmask_ctx = (RRR_res_df['RRR_deltaR2'] > 0.015) & (RRR_res_df.acronym.isin(ctx_area_list))

# only include cortical areas with >= 50 selective neurons
ctx_area_included = np.array([a for a in ctx_area_list if np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=50]) 

# get the mean abs. selectivity per cortical area
# step 1. get the time-varying coefficients for all neurons and all variables
coefs_across_time = np.array([_[:-1] for _ in RRR_res_df.loc[nis_incmask_ctx,"RRR_beta"]]) # (n_neurons, n_vars, n_timesteps)
# step 2. get the abs. selectivity for all neurons and all variables
abs_sel = np.abs(coefs_across_time).sum(2) 
# step 3. get the mean abs. selectivity per cortical area
abs_sel_a = np.asarray([abs_sel[RRR_res_df.loc[nis_incmask_ctx,"acronym"]==a].mean(0) for a in ctx_area_included])
# step 4. z-score across cortical areas, for each input variable separately
abs_sel_a = (abs_sel_a - np.mean(abs_sel_a,0))/np.std(abs_sel_a, 0) 

# prepare the anatomical connectivity data
conn_mat = pd.read_csv('../data/ctx2ctx_conn.csv', header=None).values
conn_mat = np.log(conn_mat+1e-8) 
area2i = {a: i for i, a in enumerate(ctx_area_list)}

# collect the selectivity similarity and anatomical connectivity for all pairs of cortical areas with sufficient number of selective neurons
area_pair_res = dict(anatomical_connectivity=[], acronym_pairs=[], selectivity_similarity=[])
for i in range(len(ctx_area_included)):
    for j in range(i+1, len(ctx_area_included)):
        area_pair_res['acronym_pairs'].append([ctx_area_included[i], ctx_area_included[j]])
        conn = (conn_mat[area2i[ctx_area_included[i]], area2i[ctx_area_included[j]]] + conn_mat[area2i[ctx_area_included[j]], area2i[ctx_area_included[i]]]) / 2
        area_pair_res['anatomical_connectivity'].append(conn) 
        a = abs_sel_a[i]; b = abs_sel_a[j]
        area_pair_res['selectivity_similarity'].append(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))) # cosine similarity

### Plot Fig.2d using area_pair_res['anatomical_connectivity'] and area_pair_res['selectivity_similarity']
fig, ax = plt.subplots(1,1,figsize=(3.5, 3.5))
rs, ps = spearmanr(area_pair_res['anatomical_connectivity'], area_pair_res['selectivity_similarity'])
sns.regplot(x=area_pair_res['anatomical_connectivity'], y=area_pair_res['selectivity_similarity'], ax=ax, color='C3')
ax.set_ylabel("Selectivity similarity")
ax.set_xlabel("Anatomical connectivity score")
ax.set_title(r"$\rho$=" + f"{rs:.2f}" + ", p=" + f"{ps:.1e}")

# finalize the plot
sns.despine(); plt.tight_layout()
plt.savefig("sel_sim_vs_atm_conn.pdf"); plt.close('all')



