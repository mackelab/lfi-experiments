# reproducing results of the ICML 2019 'APT' submission 

Notebooks in this folder can be used to reproduce the results of the APT submission. 
Note that [code](https://github.com/mackelab/delfi_int) development on APT (termed 'SNPE-C' within the code) continued until the last days before the submission, hence there is no single commit that represents the state of the APT code producing all the original results (indeed, not even a single repository, see [here](https://github.com/mnonnenm/delfi_int/commits/snpec_disc) and [here](https://github.com/dgreenberg/delfi_int/commits/snpec)).

The figure-generating notebooks ICML_figure_XY.ipynb load results from disk that were produced by the model-fitting and model-evaluating notebooks. 
The figure-generating notebooks can still be run on the original fits and evaluation results used for the APT submission (if moved to the notebooks_apt/results/ subfolder), or you can use the _fit.ipynb and _eval.ipynb notebooks to generate more/new results.

Apart from results produced with (private forks of the) [delfi](https://github.com/mackelab/delfi_int) package (and its surrounding repos [lfi-experiements](https://github.com/mackelab/lfi-experiments) and [lfi-models](https://github.com/mackelab/lfi-models), the APT submission also heavily made use of a slightly modified version of the [SNL](https://github.com/gpapamak/snl) package to run multiple seeds of SNL, SNPE-A, SNPE-B and SMC-ABC (with Python 2!) on the Lotka-Volterra, M/G/1 and 'Gaussian' (SLCP) simulators originally used in the SNL publication. 

To reproduce individual figure, proceed as follows:
- figure 1: [ICML_figure_1.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_1.ipynb) loads existing results from disk and chooses seeds for each algorithm that looked particularly nice. If running from scratch, it is recommended to just execute [twoMoons_illustration_figure.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_illustration_figure.ipynb)

- figure 2: for panel a), execute [SLCP_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/SLCP_fit.ipynb) and then [ICML_figure_2.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_2.ipynb). Panel b) loads [SNL](https://github.com/gpapamak/snl) results. To get APT results in c), execute [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb), but note that comparison results and plotting were also done with [SNL](https://github.com/gpapamak/snl). 

- figure 3: execute [SLCP_addedNoise_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/SLCP_addedNoise_fit.ipynb) (once per noise dimensionality m), then [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb) and [SNL_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/SNL_eval.ipynb) to compute MMDs, and plot via [ICML_figure_2.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_2.ipynb). 

- figure 4: for panel a) and b), execute [LV_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/LV_fit.ipynb) and then [ICML_figure_3.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_3.ipynb). To get APT results in c), execute [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb), but note that comparison results and plotting were also done with [SNL](https://github.com/gpapamak/snl). 

- figure 5: tbd ([ICML_figure_5.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_5.ipynb) loads pre-computed results obtained by rnn notebooks found [here](https://github.com/mackelab/lfi-experiments/tree/master/snpec) )

- figure 6: it is recommended to repeatedly execute [twoMoons_illustration_figure.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_illustration_figure.ipynb), save model fits, and then run [twoMoons_allAlgs_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_allAlgs_eval.ipynb.ipynb) followed by [ICML_figure_supp.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_supp.ipynb). The original notebook for generating the fits is found [here](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/twoMoons_allAlgs_fit.ipynb).

- figure 7: panel a) uses the same data as figure 4a). Panel b) loads [SNL](https://github.com/gpapamak/snl) results. Figure generated with [ICML_figure_supp.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_supp.ipynb).

- figure 8: To get APT results, execute [MG1_fit.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/MG1_fit.ipynb) and then [APT_eval.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/APT_eval.ipynb), but note that comparison results and plotting were done with [SNL](https://github.com/gpapamak/snl). [ICML_figure_supp.ipynb](https://github.com/mackelab/lfi-experiments/blob/master/snpec/notebooks_apt/ICML_figure_supp.ipynb) merely loads and exports into SVG and PDF format.

Remark: maximum mean discrepancies (MMDs) are stored as the *square* MMDs (see [here](https://github.com/mnonnenm/SNL_py3port/blob/master/snl/inference/diagnostics/two_sample.py#L66) for source), which is why all figure-generating code takes the square-root before plotting. 