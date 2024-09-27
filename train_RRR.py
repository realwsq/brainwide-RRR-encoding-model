import pickle
from utils import remove_space
from RRRGD_main_CV import train_model_hyper_selection

### data: {eid: {
#               "Xall": (K,T,ncoef+1), # normalized across trials, 1 expanded at the end,  
#               "yall": (K,T,N), # normalized across trials
#               "setup": dict(mean_y_TN, std_y_TN, mean_X_Tv, std_X_Tv)
#               }}
data = pickle.load(open("example_data.pk", 'rb'))
RRR_p = dict(n_comp_list=[4, 5], l2_list=[75], lr=1.0)
RRRGD_fname = remove_space(f"RRRGD_{RRR_p}")

res = train_model_hyper_selection(data, model_fname=RRRGD_fname,
                                **RRR_p)
models_split, r2s_cv_best, model = res
