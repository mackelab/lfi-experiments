#/bin/sh

COMMON='--samples 2000 --seed 1 --nb --nb-flags --no-jupyter'

cd ..
python run.py gauss testd1 ${COMMON}
python run.py gauss testd2 --sim-kwargs dim=2 ${COMMON}
python run.py gauss testd1iw --iw-loss ${COMMON}
python run.py gauss testd2iw --sim-kwargs dim=2 --iw-loss ${COMMON}
