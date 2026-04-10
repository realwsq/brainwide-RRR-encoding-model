###########################################################################
### Supplementary code for the paper:
### "Rarely categorical, highly separable representations along the cortical hierarchy"
### L. Posani*, S. Wang*, S. Muscinalli, L. Paninski$, and S. Fusi$ (2026).
### Relevant: Suppl. Fig.2d (Correlation between a region’s hierarchical position and its estimated autocorrelation timescale)
###########################################################################

### Import packages
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("../utils/")
from utils.save_and_load_data import read_Xy_4_RRR
from utils.analyze_perf_utils import load_timescale_neuron
from utils.utils import p_to_text

### Prepare data for Suppl. Fig.2d
###     Data include signal timescale of single neurons (RRR_res_df['signal_tau_neuron']))
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
area2i = {a:i for i,a in enumerate(ctx_area_list)}

### signal tau increases with hierarchy
Xy_regression = read_Xy_4_RRR(verbose=True)
# compute the signal timescale and save the result in the column "signal_tau_neuron"
RRR_res_df = load_timescale_neuron(Xy_regression, RRR_res_df) 

### Plot the relationship between signal timescale and hierarchy position of cortical areas
fig, ax = plt.subplots(1, 1, figsize=(9., 3.5*1))
results = {'region': np.array([a for a in ctx_area_included]), 
        "signal timescale (s)": np.array([np.mean(RRR_res_df.loc[(nis_incmask_ctx)&(RRR_res_df.acronym==a), "signal_tau_neuron"].values) for a in ctx_area_included]),
        "hierarchy": np.array([area2i[a] for a in ctx_area_included])}
temp = RRR_res_df[(nis_incmask_ctx)&(RRR_res_df.acronym.isin(ctx_area_included))][['acronym', "signal_tau_neuron"]]
temp['hierarchy'] = [area2i[a] for a in temp.acronym]
sns.regplot(data=temp, x='hierarchy', y="signal_tau_neuron", x_estimator=np.mean, n_boot=5000, ax=ax)
for i in range(len(results['region'])):
        ax.text(results['hierarchy'][i], results["signal timescale (s)"][i], results['region'][i],
                size=8, zorder=2, color='w', fontweight='bold')
        ax.text(results['hierarchy'][i], results["signal timescale (s)"][i], results['region'][i],
                        size=9, zorder=3)
r, p = pearsonr(results['hierarchy'], results["signal timescale (s)"])
rs, ps = spearmanr(results['hierarchy'], results["signal timescale (s)"])
print("signal timescale (s)")
print("r: ", r, "p: ", p)
print("rs: ", rs, "ps: ", ps)
ax.set_title(f'{p_to_text(ps)}  '+r"$\rho$="+f"{rs:.2f}")
ax.set_xlabel('Position in the hierarchy')
ax.set_ylabel("signal timescale (s)")
sns.despine(); plt.tight_layout(); 
plt.savefig(f"signal_timescale.pdf"); plt.close('all')