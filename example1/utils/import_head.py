
def global_params(which_areas="cortexbwm", var_types="clean", inc_type='standard'):
    gp = dict(wa=which_areas, vt=var_types, it=inc_type)
    gp['wa'] = which_areas
    if which_areas == "cortexbwm":
        gp['al_exclude'] = ['root', 'void', 'y']
    else: 
        assert False, "invalid which_area"

    if inc_type == 'standard':
        gp['X_inc'] = {"min_trials": 100, "remove_block5": True, "standardize_X": True}
        gp['y_inc'] = {'smooth_w':2., 'min_mfr':.5, 'max_sp': 0.5, "min_neurons":5,
                "transform_mfr": None, "standardize_y": True, "unit_label_min": 0.}
    elif inc_type == 'original':
        gp['X_inc'] = {"max_corr":1.0,  "min_trials": 100, "remove_block5": True, "standardize_X": True}
        gp['y_inc'] = {'smooth_w':2., 'min_mfr':.5, 'max_sp': 0.5, "min_neurons":5,
                "transform_mfr": None, "z_y": True, "unit_label_min": 0.}
    elif inc_type == 'y_fr':
        gp['X_inc'] = {"min_trials": 100, "remove_block5": True, "standardize_X": True}
        gp['y_inc'] = {'smooth_w':2., 'min_mfr':.5, 'max_sp': 0.5, "min_neurons":5,
                "transform_mfr": None, "standardize_y": False, "unit_label_min": 0.}
    else: 
        assert False, "invalid inc_type"
    
    if var_types == "clean":
        gp["vl"] = ['block', 'side', 'contrast_level', 'choice', "outcome", "wheel", "whisker_max", "lick",]
        gp['v2i'] = {'block':[0,], 'side':[1], 'contrast_level': [2], 'stimulus':[1,2], 'choice': [3], 
                     'reward': [4], 'outcome': [4], 
                    "wheel":[5], "whisker_max":[6], "lick":[7], 
                    'all': list(range(8))}

        gp['tl'] = ['block','contrast_level','choice','outcome'][::-1]
        gp['bl'] = ['wheel','whisker_max','lick'] 
    elif var_types == "body":
        gp["vl"] = ['block', 'side', 'contrast_level', 'choice', "outcome", "wheel", "whisker_max", "lick", "wheel_pos", "body"]
        gp['v2i'] = {'block':[0,], 'side':[1], 'contrast_level': [2], 'stimulus':[1,2], 'choice': [3], 
                     'reward': [4], 'outcome': [4], 
                    "wheel":[5], "whisker_max":[6], "lick":[7], "wheel_pos": [8], "body":[9], 
                    'all': list(range(10))}

        gp['tl'] = ['block','contrast_level','choice','outcome'][::-1]
        gp['bl'] = ['wheel','whisker_max','lick', 'body'] 
    else: 
        assert False, "invalid var_types"

    
    if (which_areas == "cortexbwm") and (var_types == "clean") and (inc_type == "standard"):
        gp['RRRGDglobal_p'] = dict(n_comp_list=list(range(4,7)), l2_list=[75], # 4, 75
        # gp['RRRGDglobal_p'] = dict(n_comp_list=list(range(5,6)), l2_list=[75], 
                                   stratify_by=[[0,2],[0],None],)
    elif (which_areas == "cortexbwm") and (var_types == "clean") and (inc_type == "y_fr"):
        # gp['RRRGDglobal_p'] = dict(n_comp_list=list(range(4,7)), l2_list=[75], 
        gp['RRRGDglobal_p'] = dict(n_comp_list=list(range(5,6)), l2_list=[75], 
                                   stratify_by=[[0,2],[0],None],)
    elif (which_areas == "cortexbwm") and (var_types == "clean") and (inc_type == "original"):
        gp['RRRGDglobal_p'] = dict(n_comp_list=list(range(4,6)), l2_list=[75], 
                                    lr=1., patience_ncomp = [1,2], pretrained=False,)
    else: 
        assert False, "invalid var_types"
    
    return gp

