STSS 2021
=========

This is the code for our STSS 2021 submission (Detecting Scenes in Fiction Using the Embedding Delta Signal).
To use the files, the base folder variable in the scripts must be set accordingly.


Description of the folder contents
--------------------------

* `compute_signals.py` creates the signal files, first part of the hyperparameter search
* `process_signal.py` processes the signals, second part of the hyperparamter search
* `eval.sh` evaluates the processed signals, final part of the hyperparameter search
* `get_best_params.py` shows the F1 scores of the top params
* `search_top_params.py` does the IoU evaluation on the hyperparameter pre-selection
* `svm_crossval.py` Determine the best SVM C parameter
* `get_stss_results.py` Computes the IoU of the official STSS results
* `stss_results` official STSS results, readable by the IoU script
* `README.md` this file


