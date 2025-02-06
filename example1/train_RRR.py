from example1.utils.import_head import global_params
from example1.utils.save_and_load_data import read_Xy_encoding2
from example1.utils.train_and_load_model import get_RRRglobal_res

gp_setup = dict(wa = 'cortexbwm', vt='clean', it=f"standard")
gp = global_params(which_areas=gp_setup['wa'], var_types=gp_setup['vt'], inc_type=gp_setup['it'])


Xy_regression = read_Xy_encoding2(gp, verbose=True)

get_RRRglobal_res(Xy_regression, gp)

