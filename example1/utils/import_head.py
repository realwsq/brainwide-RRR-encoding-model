
def global_params(which_areas="cortexbwm", var_types="clean", inc_type='standard'):
    gp = dict(wa=which_areas, vt=var_types, it=inc_type)
    gp['wa'] = which_areas
    if which_areas == "cortexbwm":
        # Area List to EXCLUDE (neurons from these areas will be excluded from regression)
        gp['al_exclude'] = ['root', 'void', 'y'] 
    else: 
        assert False, "invalid which_area"

    if inc_type == 'standard':
        # X, y preprocessing hyperparameters
        gp['X_inc'] = {"min_trials": 100, "remove_block5": True, "standardize_X": True}
        gp['y_inc'] = {'smooth_w':2., 'min_mfr':.5, 'max_sp': 0.5, "min_neurons":5,
                "transform_mfr": None, "standardize_y": True, "unit_label_min": 0.}
    else: 
        assert False, "invalid inc_type"

    if var_types == "clean":
        # Variable List: included input variables for the regression
        gp["vl"] = ['block', 'side', 'contrast_level', 'choice', "outcome", "wheel", "whisker_max", "lick",]
    else: 
        assert False, "invalid var_types"

    
    if (which_areas == "cortexbwm") and (var_types == "clean") and (inc_type == "standard"):
        # RRR fitting hyperParameters
        gp['RRRGDglobal_p'] = dict(n_comp_list=list(range(5,6)), l2_list=[75], 
                                   stratify_by=[[0,2],[0],None],) # first try to stratify by block, contrast_level; if not enough data, stratify by block; if still not enough data, don't stratify
            
    else: 
        assert False, "invalid var_types"
    
    return gp

