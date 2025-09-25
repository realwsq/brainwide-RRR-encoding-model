import numpy as np
import pickle, os
from utils import remove_space
from RRRGD_main_CV import train_model_hyper_selection

# data: {eid: {
#               "Xall": (#trials,#timesteps,#var+1), # input variables (normalized across trials, 1 expanded at the end)  
#               "yall": (#trials,#timesteps,#neuron), # neural activity (normalized across trials)
#               "setup": dict(
#                   mean_y_TN: (#timesteps,#neuron), # mean of neural activity across trials
#                   std_y_TN: (#timesteps,#neuron), # std of neural activity across trials 
#                   mean_X_Tv: (#timesteps,#var), # mean of input variables across trials 
#                   std_X_Tv: (#timesteps,#var), # std of input variables across trials
#                   )
#               }}
data = pickle.load(open("example0/example_data.pk", 'rb'))
RRR_p = dict(n_comp_list=[3,4,5], l2_list=[10, 25, 50,100,], lr=1.0)
RRRGD_folder = "example0/trained_model"
os.makedirs(RRRGD_folder, exist_ok=True)
RRRGD_fname = remove_space(os.path.join(RRRGD_folder, f"RRRGD_{RRR_p}"))
CV_p = dict(nsplit=3, test_size=0.3, stratify_by=[None])

res = train_model_hyper_selection(data, model_fname=RRRGD_fname,
                                  **CV_p,
                                  **RRR_p)
models_split, r2s_cv_best, model = res



### visualize the fitted results
from utils import tensor2np
from example0.utils.plot_raster import plot_single_neuron_activity
from RRRGD_main_CV import stratify_data_multi_attempts
eid = list(data.keys())[0]
# step 1: ensemble the results from all splits 
#   k=0 is the training set, 
#   k=1 is the testing set, i.e., data not seen during training
X_all, y_all, ypred_all = [], [], []
for spliti in range(CV_p['nsplit']):
    data = stratify_data_multi_attempts(data, spliti, stratify_by=CV_p['stratify_by'], test_size=CV_p['test_size'])
    X, y, ypred = models_split[spliti].predict_y_fr(data, eid, k=1)
    X_all.append(tensor2np(X))
    y_all.append(tensor2np(y))
    ypred_all.append(tensor2np(ypred))
X_all = np.concatenate(X_all, axis=0)  
y_all = np.concatenate(y_all, axis=0)
ypred_all = np.concatenate(ypred_all, axis=0)

# step 2: plot the fitted results for the top 3 neurons with the best R2
for ni in np.argsort(r2s_cv_best[eid])[-3:]:
    plot_single_neuron_activity(X_all, y_all[:,:,ni], ypred_all[:,:,ni],
                                fname=f"example0/pred_{ni}.png")

