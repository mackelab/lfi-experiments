#/bin/sh

COMMON='--samples 2000 --seed 1 --nb --no-browser'

cd ..
python run.py mog testd1 ${COMMON}
python run.py mog testd2 --sim-kwargs dim=2 ${COMMON}
python run.py mog testd1iw --iw-loss ${COMMON}
python run.py mog testd2iw --sim-kwargs dim=2 --iw-loss ${COMMON}
