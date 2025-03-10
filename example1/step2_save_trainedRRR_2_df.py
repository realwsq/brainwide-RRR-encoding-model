from example1.utils.import_head import global_params
from example1.utils.save_and_load_data import read_Xy_encoding2, load_df_from_Xy_regression_setup, get_data_folder
from example1.utils.train_and_load_model import load_RRRglobal_res
from example1.utils.analyze_perf_utils import load_xpsth_r2

import os

# folder where the dataFrame format of the trained model will be saved to
local_folder_lf = "./example1/trained_model"
fname = os.path.join(local_folder_lf, "RRRglobal_full.json")



### load the data and the trained model
# load the trained model
gp_setup = dict(wa = 'cortexbwm', vt='clean', it=f"standard")
gp = global_params(which_areas=gp_setup['wa'], var_types=gp_setup['vt'], inc_type=gp_setup['it'])
RRR_res_df = load_RRRglobal_res(gp)
# to merge the uuids, acronym, mfr_task from the data to RRR_res_df
Xy_regression = read_Xy_encoding2(gp, verbose=True)
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
