#/bin/sh

COMMON='--seed 1 --samples 2000 --nb --no-browser'

cd ..
python run.py glm testc1 --rep 2 ${COMMON}
python run.py glm testc1iw --rep 2 --iw-loss ${COMMON}
python run.py glm testrnnc1 --rnn 10 --rep 2 ${COMMON}
python run.py glm testrnnc1iw --rnn 10 --rep 2 --iw-loss ${COMMON}
