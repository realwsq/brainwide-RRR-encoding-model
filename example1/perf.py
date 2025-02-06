from example1.utils.import_head import global_params
from example1.utils.save_and_load_data import read_Xy_encoding2, load_df_from_Xy_regression_setup, get_data_folder
from example1.utils.train_and_load_model import load_RRRglobal_res
from example1.utils.analyze_perf_utils import load_xpsth_r2
from example1.utils.utils import plot_r2_comp, import_areagroup
from utils import make_folder

import pdb, os
import numpy as np

## load data
## new version
gp_setup = dict(wa = 'cortexbwm', vt='clean', it=f"original")
gp = global_params(which_areas=gp_setup['wa'], var_types=gp_setup['vt'], inc_type=gp_setup['it'])
local_folder_lf = make_folder(get_data_folder("RRR_local_folder_original"))
resgood_folder = make_folder("./example1/res_good")


Xy_regression = read_Xy_encoding2(gp, verbose=True)
data_df = load_df_from_Xy_regression_setup(['mfr_task'], Xy_regression)
RRR_res_df = load_RRRglobal_res(gp)
RRR_res_df = RRR_res_df.merge(data_df, left_on=['eid', 'ni'], right_on=['eid', 'ni'])
print(RRR_res_df.shape)
"""
local_folder_lf = make_folder(get_data_folder("RRR_local_folder_original"))
number of sessions: 178
number of ctx neruons: 14283
number of selective ctx neruons: 4617
mean R2: 0.1600486564969165
mean \Delta R2: 0.0532737727292703
mean R2 (ctx): 0.15057067475019936
mean \Delta R2 (ctx): 0.04875519381003628
"""


res = import_areagroup()
conn_area_list = res['cortical_area_list']
idxs_task = np.concatenate([gp['v2i'][v] for v in gp['tl']])
idxs_mov = np.concatenate([gp['v2i'][v] for v in gp['bl']])


inc_param=dict(min_N=50, min_r2=0.015) 
RRR_res_df = load_xpsth_r2(Xy_regression, RRR_res_df, mname="meanact", local_folder=local_folder_lf)
print(RRR_res_df.shape)
nis_incmask = (RRR_res_df["RRRglobal_r2"] - RRR_res_df['meanact_r2']) > inc_param['min_r2']
nis_incmask_ctx = nis_incmask & RRR_res_df.acronym.isin(conn_area_list)
print("number of ctx neruons:", np.sum(RRR_res_df.acronym.isin(conn_area_list)))
print("number of selective ctx neruons:", np.sum(nis_incmask_ctx))
print("mean R2:", np.mean(RRR_res_df.loc[nis_incmask, "RRRglobal_r2"]))
print("mean \Delta R2:", np.mean(RRR_res_df.loc[nis_incmask, "RRRglobal_r2"]-RRR_res_df.loc[nis_incmask, "meanact_r2"]))
print("mean R2 (ctx):", np.mean(RRR_res_df.loc[nis_incmask_ctx, "RRRglobal_r2"]))
print("mean \Delta R2 (ctx):", np.mean(RRR_res_df.loc[nis_incmask_ctx, "RRRglobal_r2"]-RRR_res_df.loc[nis_incmask_ctx, "meanact_r2"]))

RRR_res_df[['RRRglobal_r2','uuids', 'eid', 'meanact_r2']].to_csv(os.path.join(local_folder_lf, "RRRglobal_r2.csv"), index=False)
## will be used in the further analysis
RRR_res_df[['eid', 'ni', 'uuids', 
            'acronym', 'mfr_task', 
            'RRRglobal_beta', 'RRRglobal_U', 'RRRglobal_V', 'RRRglobal_b', 
            'RRRglobal_r2', 'meanact_r2',]].to_json(os.path.join(local_folder_lf, "RRRglobal_full.json"), index=False)


RRR_res_df = load_xpsth_r2(Xy_regression, RRR_res_df, mname="meanact", local_folder=local_folder_lf)
plot_r2_comp(RRR_res_df, nis_incmask, "meanact", "trial-avg", resgood_folder)

RRR_res_df = load_xpsth_r2(Xy_regression, RRR_res_df, mname="taskpsth", local_folder=local_folder_lf, idxs_psth=idxs_task)
plot_r2_comp(RRR_res_df, nis_incmask, "taskpsth", "trial-avg per task condition", resgood_folder)

pdb.set_trace()