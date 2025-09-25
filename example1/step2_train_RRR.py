from example1.utils.save_and_load_data import read_Xy_encoding2, data_params
from example1.utils.train_and_load_model import get_RRRglobal_res, RRRglobal_params


# data processing parameters
data_p = data_params(which_areas='all_but_invalid')
Xy_regression = read_Xy_encoding2(data_p, verbose=True)

# RRR fitting hyperParameters
RRRGD_p = RRRglobal_params(sample=False)
get_RRRglobal_res(Xy_regression, data_p, RRRGD_p)

