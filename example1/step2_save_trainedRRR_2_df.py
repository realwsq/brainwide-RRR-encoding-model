from example1.utils.save_and_load_data import read_Xy_encoding2, load_df_from_Xy_regression_setup, data_params
from example1.utils.train_and_load_model import load_RRRglobal_res, RRRglobal_params
from example1.utils.analyze_perf_utils import load_xpsth_r2
from utils import make_folder

import os

# folder where the dataFrame format of the trained model will be saved to
local_folder_lf = make_folder("./example1/trained_model")
fname = os.path.join(local_folder_lf, "RRRglobal_full.json")



### load the data and the trained model
# load the trained model
data_p = data_params(which_areas='all_but_invalid')
RRRGD_p = RRRglobal_params(sample=False)
RRR_res_df = load_RRRglobal_res(data_p, RRRGD_p)
# to merge the uuids, acronym, mfr_task from the data to RRR_res_df
Xy_regression = read_Xy_encoding2(data_p, verbose=True)
data_df = load_df_from_Xy_regression_setup(['mfr_task'], Xy_regression)
RRR_res_df = RRR_res_df.merge(data_df, left_on=['eid', 'ni'], right_on=['eid', 'ni'])

# compute the R2 of a baseline model that does not incorporate variables as inputs
# that is, it predicts with the mean activity of the neuron averaged across the training set
RRR_res_df = load_xpsth_r2(Xy_regression, RRR_res_df, mname="meanact")

### save the trained model in the dataFrame format
RRR_res_df[['eid', 'ni', 'uuids', 
            'acronym', 'mfr_task', 
            'RRRglobal_beta', 'RRRglobal_U', 'RRRglobal_V', 'RRRglobal_b', 
            'RRRglobal_r2', 'meanact_r2',]].to_json(os.path.join(local_folder_lf, "RRRglobal_full.json"), index=False)
