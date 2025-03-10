from example1.utils.utils import plot_r2_comp

import os
import numpy as np
import pandas as pd


# folder where the trained model is saved
local_folder_lf = "./example1/trained_model"
# load the trained model
RRR_res_df = pd.read_json(os.path.join(local_folder_lf, "RRRglobal_full.json"))
import pdb; pdb.set_trace()
# folder where the results will be saved
resgood_folder = "./example1/results"


# only include neurons whose R2 pass the minimmum \Delta R2 threshold
RRR_res_df['RRRglobal_r2inc'] = RRR_res_df["RRRglobal_r2"] - RRR_res_df['meanact_r2']
nis_incmask = RRR_res_df['RRRglobal_r2inc'] > 0.015
print("mean R2:", np.mean(RRR_res_df.loc[nis_incmask, "RRRglobal_r2"]))
print("mean \Delta R2:", np.mean(RRR_res_df.loc[nis_incmask, "RRRglobal_r2"]-RRR_res_df.loc[nis_incmask, "meanact_r2"]))

plot_r2_comp(RRR_res_df, nis_incmask, "meanact", "trial-avg", resgood_folder)
