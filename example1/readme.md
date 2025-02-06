### step 0:

- ** the processed data and trained model is saved in `get_data_folder(gp['X_inc'], gp['y_inc'], gp['vl'])` ** and `gp['X_inc'], gp['y_inc'], gp['vl']` is defined and specified in `global_params` function in `utils.import_head.py`
- the parent folder of the data (and trained model) is specified in `get_data_folder` function in `utils.save_and_load_data.py`
- the json format of the trained model is saved in `get_data_folder("RRR_local_folder_original")`, as specified in `perf.py`


### step 1:

- run `pip install ONE-api` to install ONE-api
- run `python -m example2.IBL_downloaddata` to download the data. The downloaded data will be saved in `get_data_folder('downloaded')`, as specified in `IBL_downloaddata.py`.

### step 2:

- run `python -m example2.train_RRR` to train the model. The trained model will be saved in `get_data_folder(gp['X_inc'], gp['y_inc'], gp['vl'])`.
- run `python -m example1.perf` to get the json format of the trained model. The file will be saved in `get_data_folder("RRR_local_folder_original")`, as specified in `perf.py`.

### (step 3):

- run `python -m example1.analysis_timescale` and `python -m example1.analysis_sel` to analyze the model coefficients.