#/bin/sh

COMMON='--rep 4 --samples 2000 --seed 1 --nb --no-browser'

cd ..

python run.py gauss testd1 ${COMMON}
python run.py gauss testd2 --sim-kwargs dim=2 ${COMMON}
python run.py gauss testd5 --sim-kwargs dim=5 ${COMMON}

python run.py gauss testd1iw --iw-loss ${COMMON}
python run.py gauss testd2iw --sim-kwargs dim=2 --iw-loss ${COMMON}
python run.py gauss testd5iw --sim-kwargs dim=5 --iw-loss ${COMMON}

python run.py gauss testd1svi --svi ${COMMON}
python run.py gauss testd2svi --sim-kwargs dim=2 --svi ${COMMON}
python run.py gauss testd5svi --sim-kwargs dim=5 --svi ${COMMON}

python run.py gauss testd1iwsvi --iw-loss --svi ${COMMON}
python run.py gauss testd2iwsvi --sim-kwargs dim=2 --iw-loss --svi ${COMMON}
python run.py gauss testd5iwsvi --sim-kwargs dim=5 --iw-loss --svi ${COMMON}
