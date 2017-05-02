# likelihoodfree-models

## Install and update

- Clone this repository, `cd` into it
- Run `make install`
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
  --early-stopping            If set, will do early stopping. Only works in
                              combination with validation set, i.e., make sure
                              that `--val` is greater zero.  [default: False]
  --enqueue TEXT              Enqueue the job to a given queue instead of
                              running it now. This requires a running worker
                              process, which can be started with worker.py
  --debug                     If provided, will enter debugger on error and
                              show more info during runtime.  [default: False]
  --device TEXT               Device to compute on.  [default: cpu]
  --increase-data             If set, will increase the training data on each
                              round by reloading data generated in previous
                              round.  [default: False]
  --iw-loss                   If provided, will use importance weighted loss.
                              [default: False]
  --loss-calib FLOAT          If provided, will do loss calibration with
                              Gaussian kernel centered on x0. The variance of
                              the kernel is determined by the float provided.
  --nb                        If provided, will call nb.py after fitting.
                              [default: False]
  --numerical-fix             Numerical fix (for the orginal epsilonfree
                              method).  [default: False]
  --no-browser                If provided, will not open plots of nb.py in
                              browser.  [default: False]
  --pdb-iter INTEGER          Number of iterations after which to debug.
  --prior-alpha FLOAT         If iw_loss is True, will use this alpha as
                              weight for true prior in proposal distribution.
                              [default: 0.2]
  --rep LIST OF INTEGERS      Specify the number of repetitions per
                              n_components model, seperation by comma. For
                              instance, '2,1' would mean that 2 rounds with 1
                              component are run, and 1 round with 2 components
                              are run.  [default: 2, 1]
  --rnn INTEGER               If specified, will use many-to-one RNN with
                              specified number of hidden units instead of
                              summary statistics.
  --samples LIST OF INTEGERS  Number of samples, provided as either a single
                              number or as a comma seperated list. If a list
                              is provided, say '1000,2000', 1000 samples are
                              drawn for the first round, and 2000 samples for
                              the second round. If more rounds than elements
                              in the list are run, 2000 samples will be drawn
                              for those (last list element).  [default: 2000]
  --seed INTEGER              If provided, network and simulation are seeded
  --sim-kwargs TEXT           If provided, will be passed as keyword arguments
                              to simulator. Seperate multiple keyword
                              arguments by comma, for example:
                              'duration=500,cython=True'.
  --svi                       If provided, will use SVI version  [default:
                              False]
  --train-kwargs TEXT         If provided, will be passed as keyword arguments
                              to training function (inference.train). Seperate
                              multiple keyword arguments by comma, for
                              example: 'n_iter=500,n_minibatch=200'.
  --true-prior                If provided, will use true prior on all rounds.
                              [default: False]
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
