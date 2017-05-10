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

See `run.py --help for info`


## Notes

- Make sure to have the `likelihoodfree` installed as a package, see https://github.com/mackelab/likelihoodfree


## Contributing

- Save notebooks without outputs, to save space and for better diffs; to automatically strip outputs from notebooks before committing, see https://github.com/kynan/nbstripout
