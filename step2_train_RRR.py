import os
from utils.save_and_load_data import read_Xy_4_RRR
from utils.RRRGD_main_CV import train_model_hyper_selection
from utils.utils import make_folder

model_folder = make_folder(f"./trained_RRR_model")
model_fname = os.path.join(model_folder, f"RRRGD") 
RRR_p = dict(n_comp_list=list(range(3,7)), l2_list=[25, 75, 200], 
                # stratify strategy for cross-validation:
                #   first try to stratify by block (0) and contrast_level (2); 
                #       if not enough data, stratify by block (0); 
                #       if still not enough data, don't stratify
                stratify_by=[[0,2],[0],None])
Xy_regression = read_Xy_4_RRR(verbose=True)
# trained model will be saved in the file {model_fname}.pt
train_model_hyper_selection(Xy_regression, model_fname=model_fname,
                            **RRR_p)


