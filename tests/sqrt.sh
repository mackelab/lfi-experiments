#/bin/sh

COMMON='--rep 2 --samples 1000 --seed 1'

cd ..

python run.py sqrt testd1iwsvi --iw-loss --loss-calib 0.3 --svi ${COMMON}
# python run.py sqrt testd1iwsvi --iw-loss --loss-calib 10 --svi ${COMMON}