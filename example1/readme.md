This subfolder includes a **complete** example of fitting a reduced-rank regression (RRR) model to multiple sessions of IBL brain-wide-map dataset.


Running this example from scratch requires the following steps:
1. download sessions using IBL public API
2. train the RRR model to the downloaded data
3. examine the performance of the trained model

The first two steps can take quite long (~ 1 day) to complete, therefore we attach a trained model in `example1/trained_model` to save the effort.

Next, we introduce the pipeline step-by-step. Please refer to the corresponding scripts [1] for more detailed information.


### step 1 - downloading sessions using IBL public API:

#### before running the script:
- make sure you update the paths. Two paths will be used in this step:
    - `cache_folder` in `step1_IBL_downloaddata.py`: cache folder for saving the temporary data used during downloading.
    - the parent folder of the (downloaded) data is specified in `get_data_folder` function in `utils.save_and_load_data.py`
- make sure you have ONE-api installed
    - run `pip install ONE-api` to install ONE-api

#### running the script:
- cd to the main folder and run `python -m example1.step1_IBL_downloaddata`. The downloaded data will be saved in `get_data_folder('downloaded')`, as specified in `step1_IBL_downloaddata.py`.


### step 2 - training the RRR model using our library:
#### before running the script:
- make sure you update the paths. Three paths will be used in this step:
    - the parent folder of the data (and trained model) is specified in `get_data_folder` function in `utils.save_and_load_data.py`
    - the processed data and trained model is saved in `get_data_folder(gp['X_inc'], gp['y_inc'], gp['vl'])` and `gp['X_inc'], gp['y_inc'], gp['vl']` is defined and specified in `global_params` function in `utils.import_head.py`
    - the json format of the trained model is saved in `get_data_folder("RRR_local_folder_original")`, as specified in `perf.py`
- make sure you have pytorch installed
    - run `pip install torch` to install pytorch

#### running the script:
- cd to the main folder and run `python -m example1.step2_train_RRR`. The script will first pre-process the downloaded data (see Methods of [1] for more information) and compose two matrices (X of trial-aligned, time-varying input variables (shape [n_trials, n_timesteps, n_vars]) and y of trial-aligned, time-varying neural activity (shape [n_trials, n_timesteps, n_neurons])); and then fit a RRR model given the two matrices. The optimal rank and the optimal l2 regularization penalty are selected by a 3-fold cross-validation technique across trials. The trained RRR model will be saved in `get_data_folder(gp['X_inc'], gp['y_inc'], gp['vl'])`.
- run `python -m example1.step2_save_trainedRRR_2_df` to get the dataFrame format of the trained model. The file will be saved as `example1/trained_model/RRRglobal_full.json`. Each row of the dataFrame corresponds to a neuron, and the columns include the following information. The dataFrame can be used for further analysis.
    - uuids: the unique id of the neuron
    - eid: the session id
    - acronym: the brain region of the neuron
    - RRRglobal_r2: the average of the 3-fold cross validated R2s of the RRR model
    - RRRglobal_beta: the time-vaRRRglobal_r2rying coefficients of the neuron in the RRR model. The time-varying coefficients can be seen as a matrix of the shape (n_vars+1, n_timesteps) where the first n_vars rows are the coefficients of the input variables and the last row is the bias term. The time-varying bias should be close to zero if the input variables are normalized and non-zero if not.
    - RRRglobal_U: the loadings of the temporal basis vectors. The loadings can be seen as a matrix of the shape (n_vars, n_components) where n_components is the rank of the RRR model. 
    - RRRglobal_V: the learned shared temporal basis vectors (same across all neurons). The shared temporal basis vectors can be seen as a matrix of the shape (n_components, n_timesteps).
    - RRRglobal_b: the time-varying biases of the neurons. The time-varying biases can be seen as a list of the shape (n_timesteps). The time-varying bias should be close to zero if the input variables are normalized and non-zero if not.
    - meanact_r2: the average of the 3-fold cross validated R2s of the baseline model that does not incorporate variables as inputs


### step 3 - examining the performance of the trained model:

- run `python -m example1.step3_analyze_perf` and `python -m example1.step3_analyze_selectivity` to analyze the model coefficients. The results will be saved in `example1/results`.


### References:
[1] Posani, L., Wang, S., Muscinelli, S. P., Paninski, L., & Fusi, S. (2025). Rarely categorical, always high-dimensional: how the neural code changes along the cortical hierarchy. bioRxiv, 2024-11.