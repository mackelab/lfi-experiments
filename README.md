# likelihoodfree-models

## Install and update

- Clone this repository, `cd` into it
- Run `python setup.py develop`
- Just `git pull` to update codebase; it's not necessary to run the setup again,
  since it's linked symbolically


## Usage

- Launch simulations using `run.py`, see `run.py --help` for options
- Create a notebook with plots after the run is finished, see `python nb.py --help` for options
- In addition, examples are provided in `notebooks` folder


## Example

```bash
python run.py mog quicktest --nb
```

## Options

```text
Usage: run.py [OPTIONS] MODEL PREFIX

  Run model

  Call run.py together with a prefix and a model to run.

  See run.py --help for info on parameters.

Options:
  --enqueue                   Enqueue the job rather than running it now. This
                              requires a running worker process, which can be
                              started with worker.py  [default: False]
  --debug                     If provided, will enter debugger on error.
                              [default: False]
  --device TEXT               Device to compute on.  [default: cpu]
  --iw-loss                   If provided, will use importance weighted loss.
                              [default: False]
  --nb                        If provided, will call nb.py after fitting.
                              [default: False]
  --nb-flags TEXT             If provided, will be passed to nb.py.  [default:
                              ]
  --pdb-iter INTEGER          Number of iterations after which to debug.
  --prior-alpha FLOAT         If provided, will use alpha as weight for true
                              prior in proposal distribution (only used if
                              iw_loss is True).  [default: 0.25]
  --rep LIST OF INTEGERS      Specify the number of repetitions per
                              n_components model, seperation by comma. For
                              instance, '2,1' would mean that 2 itertions with
                              1 component are run, and 1 iteration with 2
                              components are run.  [default: 2, 1]
  --rnn INTEGER               If specified, will use many-to-one RNN with
                              specified number of hidden units instead of
                              summary statistics.
  --samples LIST OF INTEGERS  Number of samples, provided as either a single
                              number or as a comma seperated list. If a list
                              is provided, say '1000,2000', 1000 samples are
                              drawn for the first iteration, and 2000 samples
                              for the second iteration. If more iterations
                              than elements in the list are run, 2000 samples
                              will be drawn for those (last list element).
                              [default: 2000]
  --seed INTEGER              If provided, network and simulation are seeded
  --sim-kwargs TEXT           If provided, will turned into dict and passed as
                              kwargs to simulator.
  --svi                       If provided, will use SVI version  [default:
                              False]
  --train-kwargs TEXT         If provided, will turned into dict and passed as
                              kwargs to inference.train.
  --true-prior                If provided, will use true prior on all
                              iterations.  [default: False]
  --units LIST OF INTEGERS    List of integers such that each list element
                              specifies the number of units per fully
                              connected hidden layer. The length of the list
                              equals the number of hidden layers.  [default:
                              50]
  --val INTEGER               Number of samples for validation.  [default: 0]
  --help                      Show this message and exit.
```

## Notes

- Make sure to have the `likelihoodfree` installed as a package, see https://github.com/mackelab/likelihoodfree


## Contributing

- Save notebooks without outputs, to save space and for better diffs; to automatically strip outputs from notebooks before committing, see https://github.com/kynan/nbstripout
