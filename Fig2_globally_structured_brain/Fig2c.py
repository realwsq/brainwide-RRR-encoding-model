###########################################################################
### Supplementary code for the paper:
### "Rarely categorical, highly separable representations along the cortical hierarchy"
### L. Posani*, S. Wang*, S. Muscinalli, L. Paninski$, and S. Fusi$ (2026).
### Relevant: Fig.2c (Average (absolute) selectivity profiles for neurons in the analyzed cortical areas)
###########################################################################

### Import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

### Prepare data for Fig.2c
###     Data include the mean absolute selectivity matrix (abs_sel_a)
# load the trained model
RRR_res_df = pd.read_json("../trained_RRR_model/RRR_selectivity.json")

# only include neurons 
#   1. whose R2 pass the minimmum \Delta R2 threshold (0.015)
#   2. that are located in cortical areas
ctx_area_list = pd.read_csv('../data/area_list.csv', header=None).values[:,0]  # list of cortical areas, ordered by hierarchy
RRR_res_df['RRR_deltaR2'] = RRR_res_df["RRR_r2"] - RRR_res_df['null_r2']
nis_incmask_ctx = (RRR_res_df['RRR_deltaR2'] > 0.015) & (RRR_res_df.acronym.isin(ctx_area_list))

# only include cortical areas with >= 30 selective neurons
ctx_area_included = np.array([a for a in ctx_area_list if np.sum((nis_incmask_ctx)&(RRR_res_df.acronym==a))>=30]) 

# order the cortical areas for plotting, based on the hierarchy of cortical areas (from sensory to associative areas)
ctx_area_ordered = [a for a in ctx_area_included if "SS" in a] + \
                    [a for a in ctx_area_included if "VIS" in a] + \
                    [a for a in ctx_area_included if ("AUD" in a) or ("GU" in a)] + \
                    [a for a in ctx_area_included if "RSP" in a] + \
                    [a for a in ctx_area_included if "AI" in a] 
ctx_area_ordered += [a for a in ctx_area_included if a not in ctx_area_ordered] 
ctx_area_ordered = ctx_area_ordered[::-1]

# get the mean abs. selectivity per cortical area
# step 1. get the time-varying coefficients for all neurons and all variables
coefs_across_time = np.array([_[:-1] for _ in RRR_res_df.loc[nis_incmask_ctx,"RRR_beta"]]) # (n_neurons, n_vars, n_timesteps)
# step 2. get the abs. selectivity for all neurons and all variables
abs_sel = np.abs(coefs_across_time).sum(2) 
# step 3. get the mean abs. selectivity per cortical area
abs_sel_a = np.asarray([abs_sel[RRR_res_df.loc[nis_incmask_ctx,"acronym"]==a].mean(0) for a in ctx_area_ordered])
# step 4. z-score across cortical areas, for each input variable separately
abs_sel_a = (abs_sel_a - np.mean(abs_sel_a,0))/np.std(abs_sel_a, 0) 

### Plot Fig.2c using the mean abs. selectivity matrix (abs_sel_a)
# input variables used in the encoding analysis
var_list = ['block', 'side', 'contrast', 'choice', "outcome", "wheel", "whisker", "lick"]
# plot the matrix
fig, ax = plt.subplots(1,1,figsize=(3.5, 5.6))
improp = dict(aspect='auto', cmap='Reds', interpolation='nearest', 
            vmax=np.max(abs_sel_a), vmin=np.min(abs_sel_a))
im = ax.imshow(abs_sel_a, **improp)
# setup of labels
_=ax.set_xticks(np.arange(abs_sel_a.shape[1]))
_=ax.set_xticklabels(var_list, rotation = 90) 
_=ax.set_yticks(np.arange(len(ctx_area_ordered)))
_=ax.set_yticklabels(ctx_area_ordered) 
# setup of colorbar
cbar = plt.colorbar(im, shrink=.4); 
_=cbar.ax.set_ylabel(r'mean $|\alpha|$ (z-scored)', fontsize=12)
cbar.ax.tick_params(labelsize=12)
# annotate the top 3 areas with the highest mean abs. selectivity for each variable
for vi in range(len(var_list)):
    _sorted = np.argsort(abs_sel_a[:,vi])[::-1]
    ais = _sorted[:3]
    for ai in ais:
        y,x = ai+0.5, vi
        _ = ax.annotate("*", xy=(x, y), 
                        xytext=(x, y),  
                        ha='center', 
                        c='k')
# finalize the plot
sns.despine(); plt.tight_layout(); 
plt.savefig("abs_sel_per_area.pdf"); plt.close('all')
