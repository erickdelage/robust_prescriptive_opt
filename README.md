# robust_prescriptive_opt

This is the repository associated with the paper titled "Robust Data-driven Prescriptiveness Optimization" published at ICML 2024. The repository includes the code used to obtain the results associated with the shortest path problem.

The main script to run is titled "Contextual_main.py". All results can be replicated after adjusting the maximum distribution shift and the version of the shortest path problem to be solved (relaxed/binary) in the "User settings" section.

Once the run is finished, the out-of-sample coefficients of prescriptiveness associated with all four methods will be saved in an Excel file titled "vhat".
Please note that "cvar_tree_utilities.py" and "tree.py" are taken from [https://github.com/CausalML/StochOptForest](https://github.com/CausalML/StochOptForest).
