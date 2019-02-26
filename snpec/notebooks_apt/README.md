# reproducing results of the ICML 2019 'APT' submission 

Notebooks in this folder can be used to reproduce results of the APT submission (but not all, see below). 

Results (except for figure 5, see below) were reproducible with [this commit](https://github.com/mnonnenm/delfi_int/commit/70d5a6701c304287098b0be34f33871fabcf1e4e). 
 The state of relevant code for most of the results at submission time was [this commit](https://github.com/mnonnenm/delfi_int/commit/3e55de7b50874f09269326ddc48dcc63d347f58a). 
Note that code development on APT continued until the last days before the submission, hence there is no single commit that represents the state of the APT code producing all the original results. 

To generate figures, run notebooks ICML_figure_XY.ipynb. These load results from disk that were produced by the model-fitting and model-evaluating notebooks (saved in the notebooks_apt/results/ subfolder)
You can also use the _fit.ipynb and _eval.ipynb notebooks to generate more/new results.

About half of the results and panels were produced with the delfi package (Python 3). The rest was produced with a [modified version](https://github.com/mnonnenm/snl) of the [SNL](https://github.com/gpapamak/snl) package (Python 2). These results are naturally not produced with the notebooks in this folder. See the panel-by-panel breakdown below.


To reproduce individual figures, proceed as follows:
- figure 1: [ICML_figure_1.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_1.ipynb) loads existing results from disk and chooses seeds for each algorithm that looked particularly nice. If running from scratch, just execute [twoMoons_illustration_figure.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_illustration_figure.ipynb)

- figure 2: for panel a), execute [SLCP_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/SLCP_fit.ipynb) and then [ICML_figure_2.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_2.ipynb). Panel b) loads SNL results. To get APT results in c), execute [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb). Comparison results and plotting of [MMDs](https://github.com/mnonnenm/snl/blob/master/plot_results_mmd_all_mean_sd.py) were done with SNL.

- figure 3: execute [SLCP_addedNoise_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/SLCP_addedNoise_fit.ipynb) (once per noise dimensionality m), then [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb) and [SNL_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/SNL_eval.ipynb) to compute MMDs, and plot via [ICML_figure_2.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_2.ipynb). 

- figure 4: for panel a) and b), execute [LV_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/LV_fit.ipynb) and then [ICML_figure_4.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_3.ipynb). To get APT results in c), execute [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb), but note that comparison results and plotting ([median distances](https://github.com/mnonnenm/snl/blob/master/plot_results_dist_all_mean_sd.py) and [negative log-probs](https://github.com/mnonnenm/snl/blob/master/plot_results_lprob_all_mean_sd.py) ) were done with SNL. 

- figure 5: tbd ([ICML_figure_5.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_5.ipynb) loads pre-computed results obtained by rnn notebooks found [here](https://github.com/mackelab/lfi-experiments/tree/master/snpec) ).

- figure 6: it is recommended to repeatedly execute [twoMoons_illustration_figure.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_illustration_figure.ipynb), save model fits, and then run [twoMoons_allAlgs_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_allAlgs_eval.ipynb.ipynb) followed by [ICML_figure_supp.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_supp.ipynb). The original notebook for generating the fits is found [here](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_allAlgs_fit.ipynb).

- figure 7: panel a) uses the same data as figure 4a). Panel b) loads SNL results. Figure composed with [ICML_figure_supp.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_supp.ipynb).

- figure 8: to get APT results, execute [MG1_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/MG1_fit.ipynb) and then [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb). Comparison results and plotting ([median distances](https://github.com/mnonnenm/snl/blob/master/plot_results_dist_all_mean_sd.py) and [negative log-probs](https://github.com/mnonnenm/snl/blob/master/plot_results_lprob_all_mean_sd.py) ) were done with SNL. Figure composed with [ICML_figure_supp.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_supp.ipynb).

Remark: maximum mean discrepancies (MMDs) are stored as the *square* MMDs (see [here](https://github.com/mnonnenm/SNL_py3port/blob/master/snl/inference/diagnostics/two_sample.py#L66) for source), which is why all figure-generating code takes the square-root before plotting. 
