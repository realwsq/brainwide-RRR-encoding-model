from example1.utils.download_data import load_data_from_pid, filter_trials_func, filter_neurons_func
from example1.utils.save_and_load_data import get_data_folder

import numpy as np
import os, glob, shutil

from one.api import ONE
from iblatlas.atlas import AllenAtlas


cache_folder = "/burg/stats/users/sw3894/ibl_cache/" # change to your own cache folder
one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', 
          cache_dir=cache_folder)

ba = AllenAtlas()
br = ba.regions

## setup for the downloaded trials
Twindow = 1.2; # sec
t_bf_stimOn = 0.3 # sec
spsdt = 10e-3  # sec
only_label1_neuron = False
data_folder = get_data_folder('downloaded')

# we want to search for probe insertions with the following areas recorded
areas_all = ["VISp", "AUDp", "SSp-ll", "AUDd", "SSp-n", "SSp-ul", "AIp",
            "SSp-m", "SSp-un", "SSp-bfd", "VISl", "AUDv", "SSs", "VISC", "SSp-tr",
            "VISli", "MOp", "VISrl", "VISpl", "RSPv", "RSPd", "GU", "RSPagl", "PERI",
            "ECT", "VISal", "ILA", "ORBl", "AId", "VISpm", "ORBm", "PL", "VISpor", "FRP",
            "AUDpo", "TEa", "VISa", "VISam", "MOs", "ORBvl", "ACAv", "ACAd", "AIv"]
for area_acronym in areas_all:
    pids_area = np.array([_ for _ in one.search_insertions(atlas_acronym=area_acronym, query_type='remote')])
    print(f"area {area_acronym} has {len(pids_area)} pids")

    # to encourage a more balanced coverage of cortical areas,
    # we will only download maximum 30 sessions per area
    _npids = 30;
    _pidi_valid = 0
    for pidi, pid_ in enumerate(pids_area):
        if _pidi_valid >= _npids:
            break

        [eid, pname] = one.pid2eid(pid_)

        fname = os.path.join(data_folder, f"data_wtonguepaw_{eid}_all_spsT{int(spsdt * 1e3)}_{only_label1_neuron}.npz")
        if os.path.isfile(fname):
            _pidi_valid += 1
            continue

        data_ret = load_data_from_pid(eid, one, ba,
                                      lambda _: filter_trials_func(_,
                                                                   remove_timeextreme_event=(True, 0.8)),
                                      lambda _: filter_neurons_func(_,
                                                                    remove_frextreme=(True, 0.5, 50.),
                                                                    only_goodneuron=(only_label1_neuron),
                                                                    only_area=(False)),
                                      spsdt=spsdt, min_neurons=10, Twindow=Twindow, t_bf_stimOn=t_bf_stimOn,
                                      load_motion_energy=True, load_wheel_velocity=True,
                                      load_tongue=True)
        if data_ret is None:
            continue

        np.savez(fname,
                 spike_count_matrix=data_ret['spike_count_matrix'],
                 behavior=data_ret['behavior'],
                 timeline=data_ret['timeline'],
                 clusters_g=data_ret['clusters_g'],
                 pid=pid_, eid=eid,
                 wheel_vel=data_ret['wheel_vel'],
                 whisker_motion=data_ret['whisker_motion'],
                 licks=data_ret["licks"],
                 )

        _pidi_valid += 1

        # delete cache
        for fname in glob.glob(os.path.join(cache_folder, "*")):
            if os.path.isdir(fname):
                shutil.rmtree(fname)
