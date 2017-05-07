#/bin/sh

COMMON='--rep 4 --samples 2000 --seed 1'

cd ..

python run.py sqrt testd1iwsvi --iw-loss --svi ${COMMON}
