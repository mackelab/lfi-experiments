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
  --enqueue                 Enqueue job rather than running it now. This
                            requires a running worker process, which can be
                            started with worker.py
  --debug / --no-debug      If True, will enter debugger on error.
  --device TEXT             Device to compute on.
  --iw-loss / --no-iw-loss  Use IW loss?
  --nb                      If provided, will call nb.py after fitting.
  --nb-flags TEXT           If provided, will be passed to nb.py.
  --pdb-iter INTEGER        Number of iterations after which to debug.
  --prior-alpha FLOAT       If provided, will use alpha as weight for true
                            prior in proposal distribution (only used if
                            iw_loss is True).
  --rep TEXT                Specify the number of repetitions per n_components
                            model, seperation by comma. For instance, '2,1'
                            would mean that 2 itertions with 1 component are
                            run, and 1 iteration with 2 components are run.
  --rnn INTEGER             If specified, will use many-to-one RNN with
                            specified number of hidden units instead of
                            summary statistics.
  --samples TEXT            Number of samples, provided as either a single
                            number or as a comma seperated list. If a list is
                            provided, say '500,2000', 500 samples are drawn
                            for the first iteration, and 2000 samples for the
                            second iteration. If more iterations are run, 2000
                            samples will be drawn (last list element).
  --seed INTEGER            If provided, network and simulation are seeded
  --sim-kwargs TEXT         If provided, will turned into dict and passed as
                            kwargs to simulator.
  --svi / --no-svi          Use SVI version?
  --train-kwargs TEXT       If provided, will turned into dict and passed as
                            kwargs to inference.train.
  --true-prior              If True, will use true prior on all iterations.
  --val INTEGER             Number of samples for validation.
  --help                    Show this message and exit.
```

## Notes

- Make sure to have the `likelihoodfree` installed as a package, see https://github.com/mackelab/likelihoodfree


## Contributing

- Save notebooks without outputs, to save space and for better diffs; to automatically strip outputs from notebooks before committing, see https://github.com/kynan/nbstripout
