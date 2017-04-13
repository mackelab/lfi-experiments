#/bin/sh

COMMON='--seed 1 --nb --no-browser'

cd ..
python run.py hh testc1 --rep 2 ${COMMON}
python run.py hh testc1iw --rep 2 --iw-loss ${COMMON}
python run.py hh testrnnc1 --rnn 20 --rep 2 ${COMMON}
python run.py hh testrnnc1iw --rnn 20 --rep 2 --iw-loss ${COMMON}
