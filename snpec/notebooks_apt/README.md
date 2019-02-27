# reproducing results of the ICML 2019 'APT' submission 

About half of the results and figure panels of the APT submission were produced with the delfi package (Python 3). The rest was produced with a [modified version](https://github.com/mnonnenm/snl) of the SNL package (Python 2). See the panel-by-panel figure breakdown below.
For a description of how to reproduce the results from the SNL package, see the bottom of this page. 


## delfi results

Notebooks in this folder can be used to reproduce delfi-generated results of the APT submission. 

To generate figures, run notebooks ICML_figure_XY.ipynb. These load results from disk that were produced by the model-fitting and model-evaluating notebooks (saved in the notebooks_apt/results/ subfolder)
You can also use the _fit.ipynb and _eval.ipynb notebooks to generate more/new results.

Results (except for figure 5, see below) were reproducible with [this commit](https://github.com/mnonnenm/delfi_int/commit/70d5a6701c304287098b0be34f33871fabcf1e4e). 
The state of relevant code for most of the results at submission time was [this commit](https://github.com/mnonnenm/delfi_int/commit/3e55de7b50874f09269326ddc48dcc63d347f58a). 
Note that code development on APT continued until the last days before the submission, hence there is no single commit that represents the state of the APT code producing all the original results. 

To reproduce individual figures, proceed as follows:
- figure 1: [ICML_figure_1.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_1.ipynb) loads existing results from disk and chooses seeds for each algorithm that looked particularly nice. If running from scratch, just execute [twoMoons_illustration_figure.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_illustration_figure.ipynb)

- figure 2: for panel a), execute [SLCP_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/SLCP_fit.ipynb) and then [ICML_figure_2.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_2.ipynb). To get APT results in c), execute [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb). For panel b) and the rest of panel c), see the SNL section below. 

- figure 3: execute [SLCP_addedNoise_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/SLCP_addedNoise_fit.ipynb) (once per noise dimensionality m), then [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb) and [SNL_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/SNL_eval.ipynb) to compute MMDs, and plot via [ICML_figure_2.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_2.ipynb). 

- figure 4: for panel a) and b), execute [LV_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/LV_fit.ipynb) and then [ICML_figure_4.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_3.ipynb). To get APT results in c), execute [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb). 
For the rest of panel c) see the SNL section below. 

- figure 5: tbd ([ICML_figure_5.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_5.ipynb) loads pre-computed results obtained by rnn notebooks found [here](https://github.com/mackelab/lfi-experiments/tree/master/snpec) ).

- figure 6: it is recommended to repeatedly execute [twoMoons_illustration_figure.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_illustration_figure.ipynb), save model fits, and then run [twoMoons_allAlgs_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_allAlgs_eval.ipynb.ipynb) followed by [ICML_figure_supp.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_supp.ipynb). The original notebook for generating the fits is found [here](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_allAlgs_fit.ipynb).

- figure 7: panel a) uses the same data as figure 4a). To produce the data for panel b) see the SNL section below. 
 Figure composed with [ICML_figure_supp.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_supp.ipynb).

- figure 8: to get APT results, execute [MG1_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/MG1_fit.ipynb) and then [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb). Comparison results and plotting ([median distances](https://github.com/mnonnenm/snl/blob/master/plot_results_dist_all_mean_sd.py) and [negative log-probs](https://github.com/mnonnenm/snl/blob/master/plot_results_lprob_all_mean_sd.py) ) were done with SNL. Figure composed with [ICML_figure_supp.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_supp.ipynb).

Remark: maximum mean discrepancies (MMDs) are stored as the *square* MMDs (see [here](https://github.com/mnonnenm/SNL_py3port/blob/master/snl/inference/diagnostics/two_sample.py#L66) for source), which is why all figure-generating code takes the square-root before plotting. 


## SNL results

### reproducing fits

The relevant SNL fork is found [here](https://github.com/mnonnenm/snl). See [here](https://github.com/gpapamak/snl/blob/master/README.md) for general usage of the package. SNL requires a Python 2 environment! 

For the results of the APT submission, the main.py file used to run the experiments was [modified](https://github.com/mnonnenm/snl/blob/master/main.py#L50) to pass a user-controlled seed to the experiment runner. Use for instance as in

``` python main.py run exps/lv_seq.txt43 ```

which starts experiments with sequential algorithms (SNPE-A, SNPE-B and SNL) on the Lotka-Volterra simulator with seed=43. The last *two* letters of the shell command will be used as seed. Experiments reproduced for the APT paper were 'lv' (Lotka-Volterra), 'gauss' (SLCP) and 'mg1' (M/G/1) each with seeds 42 to 51 (i.e. 10 different seeds). 
For SMC-ABC results on SLCP (panel 2c), additionally execute 
``` python main.py run exps/gauss_smc.txt43```
and corresponding seeds 42 to 51. Expect every experiment to take several hours per seed.

### reproducing figures

For comparing algorithm performances across seeds, use separate plotting functions to visualize [average distances](https://github.com/mnonnenm/snl/blob/master/plot_results_dist_all_mean_sd.py), [negative log-probabilities](https://github.com/mnonnenm/snl/blob/master/plot_results_lprob_all_mean_sd.py) and [maxmimum mean discrepancies](https://github.com/mnonnenm/snl/blob/master/plot_results_mmd_all_mean_sd.py) for all considered algorithms. Each takes two arguments: an experiment specifier (from 'gauss', 'lv' or 'mg1) and a specifier for APT fits. To reproduce final APT submission results, use the specifier "_validationset" for fits running SGD with stopping criterion. 

To recreate panel 2c, run 

``` python plot_results_mmd_all_mean_sd.py gauss _validationset ```

To recreate panel 4c, run 

``` python plot_results_dist_all_mean_sd.py lv _validationset ```

``` python plot_results_lprob_all_mean_sd.py lv _validationset ```

To recreate figure 8, run

``` python plot_results_dist_all_mean_sd.py mg1 _validationset ```

``` python plot_results_lprob_all_mean_sd.py mg1 _validationset ```


These commands will try to load results from disk. 
If run on experimental results other than the ones used for the original APT submission (e.g. more seeds, new specifiers for APT), make sure the expected files exist (check respective paths for [SNL](https://github.com/mnonnenm/snl/blob/master/plot_results_mmd_all_mean_sd.py#L94) and [APT](https://github.com/mnonnenm/snl/blob/master/plot_results_mmd_all_mean_sd.py#L450) results), or just out-comment lines for non-existent results as e.g. [here](https://github.com/mnonnenm/snl/blob/master/plot_results_mmd_all_mean_sd.py#L488).

When run for the first time on a new set of results, plot_results_ will first compute the respective performance metrics (neg. log-probs, distances, MMDs) and store those under data/results. This includes sampling long MCMC chains for SNL. Subsequent calls will also load performance metric results from disc.
